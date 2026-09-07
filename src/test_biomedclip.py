import sys
import torch
from configs.train_config import make_cfg
from modules import BiomedCLIPFeaturizer

def main():
    print("Testing BiomedCLIP Connection...")
    
    # 1. Mock configuration
    class DummyCfg:
        use_text_prompts = False
        shallow_layer_n = 7
        dropout = False
        projection_type = "nonlinear"
        dim = 70
        dino_feat_type = "feat"
        continuous = True
        zero_clamp = True
        pretrained_weights = None

    cfg = DummyCfg()
    
    # 2. Instantiate Featurizer
    try:
        model = BiomedCLIPFeaturizer(dim=70, cfg=cfg)
        print("✅ BiomedCLIPFeaturizer initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # 3. Test Forward Pass
    try:
        img = torch.randn(1, 3, 224, 224)
        out = model(img)
        print(f"✅ Forward pass successful. Output length: {len(out)}")
        print(f"   - Feature shape: {out[0].shape}")
        print(f"   - Code shape: {out[1].shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    print("All checks passed! BiomedCLIP is connected and functioning properly.")

if __name__ == "__main__":
    main()
