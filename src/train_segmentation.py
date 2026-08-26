import matplotlib
matplotlib.use('Agg')
import os

from torch.optim.lr_scheduler import LambdaLR
from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import seed_everything
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
import sys

torch.multiprocessing.set_sharing_strategy('file_system')

def get_class_labels(dataset_name):
    if dataset_name == "chaos":
        return ['background', 'liver', 'right kidney', 'left kidney', 'spleen']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.validation_step_outputs = []
        self.cfg = cfg
        self.n_classes = n_classes

        if getattr(cfg, "use_text_prompts", False):
            dim = 512
        elif not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cut_model = load_model(cfg.model_type, data_dir).to(device)
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        elif cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        elif cfg.arch == "biomedclip":
            self.net = BiomedCLIPFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        if getattr(cfg, "use_text_prompts", False):
            if cfg.chaos_modality == "CT":
                modality = "CT"
            else:
                modality = "MRI"
            
            class_labels = get_class_labels(cfg.dataset_name)
            prompts = []
            for label in class_labels:
                if label == 'background':
                    prompts.append(f"Abdominal background tissue in a {modality} scan")
                else:
                    prompts.append(f"A {modality} scan showing the {label}")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            text_embs = self.net.get_text_embeddings(prompts, device).detach()
            self.register_buffer('text_embeddings', text_embs)
        
        else:
            self.text_embeddings = None

        # Cluster probes take dim+2 when (x,y) spatial coordinates are appended
        self.spatial_weight = getattr(cfg, 'spatial_weight', 0.0)
        self.cluster_alpha = getattr(cfg, 'cluster_alpha', 1.0)
        cluster_dim = dim + 2 if self.spatial_weight > 0 else dim
        entropy_weight = getattr(cfg, 'cluster_entropy_weight', 0.0)
        self.train_cluster_probe = ClusterLookup(cluster_dim, n_classes, entropy_weight)
        self.cluster_probe = ClusterLookup(
            cluster_dim, n_classes + cfg.extra_clusters, entropy_weight)
        
        if self.text_embeddings is not None:
            self.train_cluster_probe.init_from(self.text_embeddings)
            # Only initialize the first n_classes slots; extra clusters stay random
            self.cluster_probe.init_from(self.text_embeddings[:n_classes])

        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)

        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", n_classes, 0, False)

        # Foreground-boosted class weights to combat background dominance
        fg_weight = getattr(cfg, 'linear_fg_weight', 3.0)
        class_weights = [1.0] + [fg_weight] * (n_classes - 1)
        self.register_buffer('linear_probe_weights', torch.tensor(class_weights))
        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss(weight=self.linear_probe_weights)
        
        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift)

        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.automatic_optimization = False

        if self.cfg.dataset_name == "chaos":
            self.label_cmap = create_chaos_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0
        self.save_hyperparameters()

    def _add_spatial_coords(self, code):
        """Concatenate weighted (x, y) coordinate channels to feature code.
        
        This gives the cluster probe spatial awareness — critical for medical
        imaging where organs are distinguished by position, not appearance.
        """
        if self.spatial_weight <= 0:
            return code
        B, C, H, W = code.shape
        coords_h = torch.linspace(-1, 1, H, device=code.device, dtype=code.dtype)
        coords_w = torch.linspace(-1, 1, W, device=code.device, dtype=code.dtype)
        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing="ij")
        # [1, 2, H, W] → expand to [B, 2, H, W]
        coords = torch.stack([grid_h, grid_w], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        coords = coords * self.spatial_weight
        return torch.cat([code, coords], dim=1)  # [B, C+2, H, W]



    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()

        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        # Unpack the CombinedLoader batches
        main_batch = batch["main"]
        neg_batch = batch["neg"]

        with torch.no_grad():
            ind = main_batch["ind"]
            img = main_batch["img"]
            img_aug = main_batch["img_aug"]
            coord_aug = main_batch["coord_aug"]
            img_pos = main_batch["img_pos"]
            label = main_batch["label"]
            label_pos = main_batch["label_pos"]
            
            neg_img = neg_batch["img"]

        feats, code = self.net(img)
        if self.cfg.correspondence_weight > 0:
            feats_pos, code_pos = self.net(img_pos)
            neg_feats, neg_code = self.net(neg_img)
        log_args = dict(sync_dist=False, rank_zero_only=True)

        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
            signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        else:
            signal = feats
            signal_pos = feats_pos if self.cfg.correspondence_weight > 0 else None

        loss = 0

        should_log_hist = (self.cfg.hist_freq is not None) and \
                          (self.global_step % self.cfg.hist_freq == 0) and \
                          (self.global_step > 0)
        if self.cfg.use_salience:
            salience = main_batch["mask"].to(torch.float32).squeeze(1)
            salience_pos = main_batch["mask_pos"].to(torch.float32).squeeze(1)
        else:
            salience = None
            salience_pos = None

        if self.cfg.correspondence_weight > 0:
            (
                pos_intra_loss, pos_intra_cd,
                pos_inter_loss, pos_inter_cd,
                neg_inter_loss, neg_inter_cd,
            ) = self.contrastive_corr_loss_fn(
                signal, signal_pos,
                salience, salience_pos,
                code, code_pos,
                neg_feats, neg_code,
            )

            if should_log_hist:
                self.logger.experiment.add_histogram("intra_cd", pos_intra_cd, self.global_step)
                self.logger.experiment.add_histogram("inter_cd", pos_inter_cd, self.global_step)
                self.logger.experiment.add_histogram("neg_cd", neg_inter_cd, self.global_step)
            neg_inter_loss = neg_inter_loss.mean()
            pos_intra_loss = pos_intra_loss.mean()
            pos_inter_loss = pos_inter_loss.mean()
            self.log('loss/pos_intra', pos_intra_loss, **log_args)
            self.log('loss/pos_inter', pos_inter_loss, **log_args)
            self.log('loss/neg_inter', neg_inter_loss, **log_args)
            self.log('cd/pos_intra', pos_intra_cd.mean(), **log_args)
            self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args)
            self.log('cd/neg_inter', neg_inter_cd.mean(), **log_args)

            loss += (self.cfg.pos_inter_weight * pos_inter_loss +
                     self.cfg.pos_intra_weight * pos_intra_loss +
                     self.cfg.neg_inter_weight * neg_inter_loss) * self.cfg.correspondence_weight

        if self.cfg.rec_weight > 0:
            rec_feats = self.decoder(code)
            rec_loss = -(norm(rec_feats) * norm(feats)).sum(1).mean()
            self.log('loss/rec', rec_loss, **log_args)
            loss += self.cfg.rec_weight * rec_loss

        if self.cfg.aug_alignment_weight > 0:
            orig_feats_aug, orig_code_aug = self.net(img_aug)
            downsampled_coord_aug = resize(
                coord_aug.permute(0, 3, 1, 2),
                orig_code_aug.shape[2]).permute(0, 2, 3, 1)
            aug_alignment = -torch.einsum(
                "bkhw,bkhw->bhw",
                norm(sample(code, downsampled_coord_aug)),
                norm(orig_code_aug)
            ).mean()
            self.log('loss/aug_alignment', aug_alignment, **log_args)
            loss += self.cfg.aug_alignment_weight * aug_alignment

        crf_weight = getattr(self.cfg, 'crf_weight', 0.5)
        if crf_weight > 0:
            crf = self.crf_loss_fn(
                resize(img, 56),
                norm(resize(code, 56))
            ).mean()
            self.log('loss/crf', crf, **log_args)
            loss += crf_weight * crf

        if getattr(self.cfg, "use_text_prompts", False) and getattr(self, "text_embeddings", None) is not None:
            normed_code = F.normalize(code, dim=1)
            normed_text = F.normalize(self.text_embeddings, dim=1).to(code.device)
            sim = torch.einsum("bchw,nc->bnhw", normed_code, normed_text) / 0.07
            soft_probs = F.softmax(sim, dim=1)
            entropy = -(soft_probs * torch.log(soft_probs + 1e-6)).sum(dim=1)
            text_align_loss = entropy.mean()
            self.log('loss/text_align', text_align_loss, **log_args)
            loss += getattr(self.cfg, "text_align_weight", 0.3) * text_align_loss

        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        detached_code = torch.clone(code.detach())

        linear_logits = self.linear_probe(detached_code)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        code_with_pos = self._add_spatial_coords(detached_code)
        cluster_loss, cluster_probs = self.cluster_probe(code_with_pos, alpha=self.cluster_alpha)
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)

        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()

        # Step all LR schedulers (net cosine decay + cluster warmup)
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if isinstance(schedulers, list):
                for sched in schedulers:
                    sched.step()
            else:
                schedulers.step()


        if self.global_step % 2000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            # Make a new tfevent file
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        return loss

    def on_train_start(self):
        tb_metrics = {
            **self.linear_metrics.compute(),
            **self.cluster_metrics.compute()
        }
        self.logger.log_hyperparams(self.cfg, tb_metrics)

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats, code = self.net(img)
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            linear_preds = self.linear_probe(code)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            code_with_pos = self._add_spatial_coords(code)
            cluster_loss, cluster_preds = self.cluster_probe(code_with_pos, None)
            cluster_preds = cluster_preds.argmax(1)

            # ── Cluster Probe Debugging ──────────────────────────────────
            if batch_idx == 0 and (self.current_epoch % 5 == 0 or self.current_epoch <= 2):
                flat_cpreds = cluster_preds.reshape(-1).cpu()
                flat_labels = label.reshape(-1).cpu()

                # 1) Unique cluster IDs and pixel counts
                unique_ids, counts = torch.unique(flat_cpreds, return_counts=True)
                print("\n" + "=" * 70)
                print(f"[CLUSTER PROBE DEBUG]  epoch={self.current_epoch}  "
                      f"step={self.global_step}  n_classes={self.n_classes}  "
                      f"extra_clusters={self.cfg.extra_clusters}")
                print(f"  Total clusters in probe: "
                      f"{self.n_classes + self.cfg.extra_clusters}")
                print(f"  Unique cluster IDs found: {unique_ids.tolist()}")
                print(f"  Pixel counts per cluster:")
                for cid, cnt in zip(unique_ids.tolist(), counts.tolist()):
                    pct = 100.0 * cnt / flat_cpreds.numel()
                    print(f"    cluster {cid:>2d}: {cnt:>8d} pixels  ({pct:5.1f}%)")

                # 2) Cross-tabulation: cluster_id × ground-truth label
                valid = (flat_labels >= 0) & (flat_labels < self.n_classes)
                if valid.any():
                    vl = flat_labels[valid]
                    vc = flat_cpreds[valid]
                    n_total_clusters = self.n_classes + self.cfg.extra_clusters
                    cross = torch.zeros(n_total_clusters, self.n_classes,
                                        dtype=torch.int64)
                    for c_id in range(n_total_clusters):
                        mask_c = (vc == c_id)
                        for l_id in range(self.n_classes):
                            cross[c_id, l_id] = (mask_c & (vl == l_id)).sum()

                    class_names = get_class_labels(self.cfg.dataset_name)[
                                  :self.n_classes]
                    header = "  cluster \\ label | " + " | ".join(
                        f"{n:>12s}" for n in class_names)
                    print(f"\n  Cross-tab (cluster × ground-truth):")
                    print(f"  {header}")
                    print(f"  {'-' * len(header)}")
                    for c_id in range(n_total_clusters):
                        row = " | ".join(
                            f"{cross[c_id, l].item():>12d}"
                            for l in range(self.n_classes))
                        tag = " (extra)" if c_id >= self.n_classes else ""
                        print(f"  cluster {c_id:>2d}{tag:>8s} | {row}")

                    # 3) Per ground-truth class: dominant cluster
                    print(f"\n  Per-class dominant cluster:")
                    for l_id, name in enumerate(class_names):
                        col = cross[:, l_id]
                        total = col.sum().item()
                        if total > 0:
                            dom = col.argmax().item()
                            dom_pct = 100.0 * col[dom].item() / total
                            print(f"    {name:>15s}: dominant cluster={dom}  "
                                  f"({dom_pct:.1f}% of {total} pixels)")
                        else:
                            print(f"    {name:>15s}: no pixels")

                # 4) Cluster centroid norms (health check)
                cnorms = self.cluster_probe.clusters.data.norm(dim=1).cpu()
                print(f"\n  Cluster centroid norms: "
                      f"{[f'{v:.3f}' for v in cnorms.tolist()]}")
                print("=" * 70 + "\n")
            # ── End Cluster Probe Debugging ───────────────────────────────

            self.cluster_metrics.update(cluster_preds, label)

            self.validation_step_outputs.append({
                'img': img[:self.cfg.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()})

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(),
                **self.cluster_metrics.compute(),
            }

            if self.trainer.is_global_zero and not self.cfg.submitting_to_aml:
                import random
                output_num = random.randint(0, len(self.validation_step_outputs) - 1)
                output = {k: v.detach().cpu() for k, v in self.validation_step_outputs[output_num].items()}

                fig, ax = plt.subplots(4, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 4 * 3))
                for i in range(self.cfg.n_images):
                    ax[0, i].imshow(prep_for_plot(output["img"][i]))
                    ax[1, i].imshow(self.label_cmap[output["label"][i]])
                    ax[2, i].imshow(self.label_cmap[output["linear_preds"][i]])
                    ax[3, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
                ax[0, 0].set_ylabel("Image", fontsize=16)
                ax[1, 0].set_ylabel("Label", fontsize=16)
                ax[2, 0].set_ylabel("Linear Probe", fontsize=16)
                ax[3, 0].set_ylabel("Cluster Probe", fontsize=16)
                remove_axes(ax)
                plt.tight_layout()
                add_plot([l.experiment for l in self.loggers], "plot_labels", self.global_step)

                if self.cfg.has_labels:
                    fig = plt.figure(figsize=(13, 10))
                    ax = fig.gca()
                    hist = self.cluster_metrics.histogram.detach().cpu().to(torch.float32)
                    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
                    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues")
                    ax.set_xlabel('Predicted labels')
                    ax.set_ylabel('True labels')
                    # Use only the first n_classes labels, then add "Extra" if needed
                    names = get_class_labels(self.cfg.dataset_name)[:self.n_classes]
                    if self.cfg.extra_clusters:
                        names = names + ["Extra"]
                    # Derive tick count from actual histogram dimensions
                    n_rows, n_cols = hist.shape
                    ax.set_xticks(np.arange(0, n_rows) + .5)
                    ax.set_yticks(np.arange(0, n_cols) + .5)
                    ax.xaxis.tick_top()
                    # Truncate names to fit histogram dimensions
                    row_names = names[:n_rows]
                    col_names = names[:n_cols]
                    ax.xaxis.set_ticklabels(row_names, fontsize=14)
                    ax.yaxis.set_ticklabels(col_names, fontsize=14)
                    colors = [self.label_cmap[i] / 255.0 if i < len(self.label_cmap)
                              else np.array([0.5, 0.5, 0.5]) for i in range(max(n_rows, n_cols))]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    ax.vlines(np.arange(0, n_rows + 1), color=[.5, .5, .5], *ax.get_xlim())
                    ax.hlines(np.arange(0, n_cols + 1), color=[.5, .5, .5], *ax.get_ylim())
                    plt.tight_layout()
                    add_plot([l.experiment for l in self.loggers], "conf_matrix", self.global_step)

                    raw_hist = self.cluster_metrics.histogram.detach().cpu().to(torch.float32)
                    all_bars = torch.cat([
                        raw_hist.sum(0),
                        raw_hist.sum(1)
                    ], axis=0)
                    ymin = max(all_bars.min() * .8, 1)
                    ymax = all_bars.max() * 1.2

                    fig, ax = plt.subplots(1, 2, figsize=(2 * 5, 1 * 4))
                    bar_data_0 = raw_hist.sum(0)
                    bar_names_0 = col_names[:len(bar_data_0)]
                    bar_colors_0 = colors[:len(bar_data_0)]
                    ax[0].bar(range(len(bar_data_0)),
                              bar_data_0,
                              tick_label=bar_names_0,
                              color=bar_colors_0)
                    ax[0].set_ylim(ymin, ymax)
                    ax[0].set_title("Label Frequency")
                    ax[0].set_yscale('log')
                    ax[0].tick_params(axis='x', labelrotation=90)

                    bar_data_1 = raw_hist.sum(1)
                    bar_names_1 = row_names[:len(bar_data_1)]
                    bar_colors_1 = colors[:len(bar_data_1)]
                    ax[1].bar(range(len(bar_data_1)),
                              bar_data_1,
                              tick_label=bar_names_1,
                              color=bar_colors_1)
                    ax[1].set_ylim(ymin, ymax)
                    ax[1].set_title("Cluster Frequency")
                    ax[1].set_yscale('log')
                    ax[1].tick_params(axis='x', labelrotation=90)

                    plt.tight_layout()
                    add_plot([l.experiment for l in self.loggers], "label frequency", self.global_step)

            if self.global_step > 2:
                self.log_dict(tb_metrics)

                if self.trainer.is_global_zero and self.cfg.azureml_logging:
                    from azureml.core.run import Run
                    run_logger = Run.get_context()
                    for metric, value in tb_metrics.items():
                        run_logger.log(metric, value)

            self.linear_metrics.reset()
            self.cluster_metrics.reset()
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        main_params = list(self.net.parameters())

        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(
            list(self.cluster_probe.parameters()),
            lr=getattr(self.cfg, 'cluster_lr', 5e-3))

        # Cosine annealing for the main network to prevent late-stage feature drift
        from torch.optim.lr_scheduler import CosineAnnealingLR
        net_scheduler = CosineAnnealingLR(net_optim, T_max=self.cfg.max_steps, eta_min=1e-6)

        # Linear warmup for cluster probe: ramp from 0 → full LR
        warmup_steps = getattr(self.cfg, 'cluster_warmup_steps', 100)
        def warmup_fn(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        cluster_scheduler = LambdaLR(cluster_probe_optim, lr_lambda=warmup_fn)

        return (
            [net_optim, linear_probe_optim, cluster_probe_optim],
            [net_scheduler, cluster_scheduler],
        )


@hydra.main(config_path="configs", config_name="train_config.yaml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=15),
        T.RandomResizedCrop(size=cfg.res, scale=(0.5, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.4, contrast=.4, saturation=.1, hue=0.0),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True
    )

    if cfg.dataset_name == "chaos":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    val_dataset = ContrastiveSegDataset(
    pytorch_data_dir=pytorch_data_dir,
    dataset_name=cfg.dataset_name,
    crop_type=None,
    image_set="val",
    transform=get_transform(cfg.res, False, val_loader_crop),
    target_transform=get_transform(cfg.res, True, val_loader_crop),
    mask=True,
    cfg=cfg,
    )

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=False)

    neg_dataset = NegativeImageDataset(
        root_dir=cfg.neg_data_dir,
        transform=get_transform(cfg.res, False, cfg.loader_crop_type)
    )
    # Use drop_last=True so that we don't crash if the batch sizes don't perfectly align
    neg_loader = DataLoader(neg_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True, pin_memory=False)

    # PyTorch Lightning natively handles dicts of dataloaders for multiple dataloaders
    combined_train_loader = {"main": train_loader, "neg": neg_loader}

    if cfg.submitting_to_aml:
        val_batch_size = 16
    else:
        val_batch_size = cfg.batch_size

    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    tb_logger = TensorBoardLogger(
        join(log_dir, name),
        default_hp_metric=False
    )
    wandb_logger = WandbLogger(
        name=name,
        project="STEGO",
        save_dir=log_dir
    )

    accelerator_type = 'gpu' if torch.cuda.is_available() else 'cpu'
    if cfg.submitting_to_aml:
        gpu_args = dict(accelerator=accelerator_type, devices=1, val_check_interval=250)

        if gpu_args["val_check_interval"] > len(train_loader):
            gpu_args.pop("val_check_interval")

    else:
        gpu_args = dict(accelerator=accelerator_type, devices=1, val_check_interval=cfg.val_freq)

        if gpu_args["val_check_interval"] > len(train_loader):
            gpu_args.pop("val_check_interval")

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=[tb_logger, wandb_logger],
        max_steps=cfg.max_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, name),
                filename="checkpoint-{epoch:02d}-{step:04d}-mIoU={test/cluster/mIoU:.4f}",
                every_n_train_steps=100,
                save_top_k=-1,
                monitor="test/cluster/mIoU",
                mode="max",
            )
        ],
        **gpu_args
    )
    trainer.fit(model, combined_train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()