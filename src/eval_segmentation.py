from modules import *
from data import *
from collections import defaultdict
from multiprocessing import Pool
import hydra
import torch.multiprocessing
from crf import batched_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_segmentation import LitUnsupervisedSegmenter

import os
from os.path import join

import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

torch.serialization.add_safe_globals([ModelCheckpoint])

def plot_cm(histogram, label_cmap, cfg):
    names = get_class_labels(cfg.dataset_name)
    if cfg.extra_clusters:
        names = names + ["Extra"]
    plot_confusion_matrix(histogram, label_cmap, names, 'Predicted labels', 'True labels')


@hydra.main(config_path="configs", config_name="eval_config.yaml")
def my_app(cfg: DictConfig) -> None:

    pytorch_data_dir = cfg.pytorch_data_dir
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)

    prepare_output_dirs(result_dir, "img", "label", "cluster")

    for model_path in cfg.model_paths:
      
        torch.multiprocessing.set_sharing_strategy('file_system')
        model = LitUnsupervisedSegmenter.load_from_checkpoint(
            model_path,
            map_location="cpu",
            weights_only=False
        )
        print(OmegaConf.to_yaml(model.cfg))

        loader_crop = "center"

        test_dataset = ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name=model.cfg.dataset_name,
            crop_type=None,
            image_set="val",
            transform=get_transform(cfg.res, False, loader_crop),
            target_transform=get_transform(cfg.res, True, loader_crop),
            mask=True,
            cfg=model.cfg,
        )

        test_loader = DataLoader(
            test_dataset,
            cfg.batch_size * 2,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=flexible_collate
        )

        device = get_device()
        model.eval().to(device)

        par_model = maybe_data_parallel(model.net, cfg.use_ddp)

        saved_data = defaultdict(list)

        with Pool(cfg.num_workers + 5) as pool:
            for i, batch in enumerate(tqdm(test_loader)):

                with torch.no_grad():
                    img = batch["img"].to(device)
                    label = batch["label"].to(device)

                    # Flip-averaged test-time augmentation, resized to match label
                    code = flip_averaged_code(par_model, img, label.shape[-2:])

                    # Predictions
                    linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)

                    # CHAOS cluster format
                    cluster_loss, cluster_probs = model.cluster_probe(code, None)
                    cluster_probs = torch.log_softmax(cluster_probs, dim=1)

                    if cfg.run_crf:
                        linear_preds = batched_crf(pool, img, linear_probs).argmax(1).cpu()
                        cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cpu()
                    else:
                        linear_preds = linear_probs.argmax(1)
                        cluster_preds = cluster_probs.argmax(1)

                    model.test_linear_metrics.update(linear_preds, label)
                    model.test_cluster_metrics.update(cluster_preds, label)

                    saved_data["linear_preds"].append(linear_preds.cpu())
                    saved_data["cluster_preds"].append(cluster_preds.cpu())
                    saved_data["label"].append(label.cpu())
                    saved_data["img"].append(img.cpu())

        saved_data = {k: torch.cat(v, dim=0) for k, v in saved_data.items()}

        tb_metrics = {
            **model.test_linear_metrics.compute(),
            **model.test_cluster_metrics.compute(),
        }

        print("")
        print(model_path)
        print(tb_metrics)

        # Save ALL images (CHAOS-friendly)
        for i in range(len(saved_data["img"])):

            plot_img = (prep_for_plot(saved_data["img"][i]) * 255).numpy().astype(np.uint8)
            plot_label = (model.label_cmap[saved_data["label"][i]]).astype(np.uint8)

            Image.fromarray(plot_img).save(join(result_dir, "img", f"{i}.jpg"))
            Image.fromarray(plot_label).save(join(result_dir, "label", f"{i}.png"))

            plot_cluster = model.label_cmap[
                model.test_cluster_metrics.map_clusters(saved_data["cluster_preds"][i])
            ].astype(np.uint8)

            Image.fromarray(plot_cluster).save(join(result_dir, "cluster", f"{i}.png"))

        plot_cm(model.test_cluster_metrics.histogram, model.label_cmap, model.cfg)
        plt.show()


if __name__ == "__main__":
    prep_args()
    my_app()