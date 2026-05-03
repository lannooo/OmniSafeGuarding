import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

class ClipAnnoySearcher:
    """
    CLIP + Annoy multimodal semantic retrieval system (no-cache version)
    Core features:
    - Build an index from an image list file
    - Persist/load index and metadata
    - Text-to-image semantic search
    """
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 annoy_metric: str = 'angular',
                 annoy_trees: int = 20,
                 device: str = None):
        """
        Initialize the retrieval system.
        
        Args:
            model_name: Hugging Face CLIP model name
            annoy_metric: Annoy distance metric ('angular' recommended)
            annoy_trees: Number of Annoy trees (more = better accuracy but slower)
            device: Device ('cuda', 'mps', 'cpu')
        """
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Initialize CLIP
        print(f"🧠 Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.feature_dim = self.model.config.projection_dim
        
        # Initialize Annoy
        self.annoy_metric = annoy_metric
        self.annoy_trees = annoy_trees
        from annoy import AnnoyIndex

        self.index = AnnoyIndex(self.feature_dim, annoy_metric)
        
        # Metadata management
        self.image_paths = []
        self.id_to_path = {}  # {item_id: image_path}
        self.path_to_id = {}  # {image_path: item_id}
        self.index_path = None
    
    def _extract_image_features(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        """
        Extract image features in batches.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size
        
        Returns:
            Normalized feature matrix [n_images, feature_dim]
        """
        features = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        print(f"🖼️ Processing {len(image_paths)} images (batch_size={batch_size})")
        for i in tqdm(range(0, len(image_paths), batch_size), total=total_batches, desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = [Image.open(path).convert("RGB") for path in batch_paths]

            inputs = self.processor(
                images=batch_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)
            
            # Normalize (critical)
            batch_features = batch_features / batch_features.norm(p=2, dim=-1, keepdim=True)
            batch_features = batch_features.cpu().numpy()
            features.append(batch_features)
        
        if not features:
            raise ValueError("No valid image features were extracted!")
        
        features = np.vstack(features)
        print(f"✅ Extraction complete: {features.shape[0]} valid features, dim={features.shape[1]}")
        return features
    
    def build_index_from_list(self,
                              image_files:list,
                              batch_size: int = 32) -> None:
        """
        Build index from an image file list.
        
        Args:
            image_files: Image file list
            batch_size: Batch size for feature extraction
        """
        self.image_paths = image_files
        
        # Extract features
        features = self._extract_image_features(self.image_paths, batch_size)
        
        # Validate feature count
        assert features.shape[0] == len(self.image_paths), "Feature count does not match image count!"
        
        # Build Annoy index
        print(f"🌳 Building Annoy index (metric={self.annoy_metric}, trees={self.annoy_trees})")
        
        # Add vectors and maintain mappings
        for item_id, (path, feature) in enumerate(zip(self.image_paths, features)):
            self.index.add_item(item_id, feature)
            self.id_to_path[item_id] = path
            self.path_to_id[path] = item_id
        
        # Build index
        self.index.build(self.annoy_trees)
        print(f"✅ Index built successfully! Total images: {len(self.image_paths)}")
        
    
    def build_index_from_file(self, 
                             image_list_file: str, 
                             batch_size: int = 32) -> None:
        """
        Build index from an image list file.
        
        Args:
            image_list_file: Text file containing image paths (one path per line)
            batch_size: Batch size for feature extraction
        """
        # Read image paths
        print(f"📄 Reading image list: {image_list_file}")
        valid_paths = []
        with open(image_list_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path and os.path.exists(path):
                    valid_paths.append(path)
                elif path:  # Path does not exist
                    print(f"⚠️ Skipping missing path: {path}")
        
        if not valid_paths:
            raise ValueError("No valid image paths were found!")
        
        print(f"🔍 Found {len(valid_paths)} valid image paths")
        self.build_index_from_list(valid_paths, batch_size)
        
    
    def save_index(self, index_path: str) -> None:
        """
        Save index to disk.
        
        Args:
            index_path: Index file path (will create both .ann and .meta files)
        """
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        
        # Save Annoy index
        ann_path = f"{index_path}.ann"
        self.index.save(ann_path)
        
        # Save metadata
        meta_path = f"{index_path}.meta"
        metadata = {
            "image_paths": self.image_paths,
            "id_to_path": self.id_to_path,
            "path_to_id": self.path_to_id,
            "feature_dim": self.feature_dim,
            "annoy_metric": self.annoy_metric,
            "model_name": self.model.name_or_path,
            "annoy_trees": self.index.get_n_trees() if hasattr(self.index, 'get_n_trees') else self.annoy_trees
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.index_path = index_path
        print(f"💾 Index saved to: {ann_path} + {meta_path}")
        print(f"📊 Index stats: {len(self.image_paths)} images, dim={self.feature_dim}")
    
    def load_index(self, index_path: str) -> None:
        """
        Load index from disk.
        
        Args:
            index_path: Index file path (requires both .ann and .meta files)
        """
        ann_path = f"{index_path}.ann"
        meta_path = f"{index_path}.meta"
        
        if not os.path.exists(ann_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Index files not found! Expected {ann_path} and {meta_path}")
        
        # Load metadata
        print(f"📂 Loading metadata: {meta_path}")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Validate model compatibility
        if metadata["model_name"] != self.model.name_or_path:
            print(f"⚠️ Warning: current model ({self.model.name_or_path}) differs from index model ({metadata['model_name']})")
        
        # Restore state
        self.image_paths = metadata["image_paths"]
        self.id_to_path = {int(k): v for k, v in metadata["id_to_path"].items()}  # Ensure integer keys
        self.path_to_id = metadata["path_to_id"]
        self.feature_dim = metadata["feature_dim"]
        loaded_trees = metadata.get("annoy_trees", self.annoy_trees)
        
        # Recreate Annoy index
        print(f"🌳 Recreating Annoy index (metric={metadata['annoy_metric']}, trees={loaded_trees})")
        self.index = AnnoyIndex(self.feature_dim, metadata["annoy_metric"])
        self.index.load(ann_path)
        
        self.index_path = index_path
        print(f"✅ Index loaded successfully! Total images: {len(self.image_paths)}")
    
    def search(self, query_text: str, k: int = 5) -> list:
        """
        Search similar images by text query.
        
        Args:
            query_text: Query text
            k: Number of results to return
        
        Returns:
            A list of dicts: {
                "path": image path,
                "similarity": cosine similarity (0~1),
                "distance": raw distance,
                "rank": rank
            }
        """
        if not self.index_path and len(self.image_paths) == 0:
            raise RuntimeError("Index is not built or loaded! Call build_index_from_file() or load_index() first")
        
        # Truncate long text (CLIP supports up to 77 tokens)
        max_length = 77
        if len(query_text.split()) > max_length:
            query_text = " ".join(query_text.split()[:max_length])
            print(f"✂️ Text truncated to {max_length} tokens")
        
        # Extract text features
        inputs = self.processor(
            text=[query_text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        # Normalize
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        query_vector = text_features[0].cpu().numpy()
        
        # Search (automatically handles k greater than index size)
        actual_k = min(k, len(self.image_paths))
        item_ids, distances = self.index.get_nns_by_vector(
            query_vector, 
            actual_k, 
            include_distances=True
        )
        
        # Convert results
        results = []
        for rank, (item_id, dist) in enumerate(zip(item_ids, distances)):
            # Convert angular distance to cosine similarity
            if self.annoy_metric == 'angular':
                similarity = 1 - (dist ** 2) / 2
            else:  # Generic conversion for other metrics
                similarity = max(0.0, 1 - dist)
            
            # Ensure item_id is int (handles string keys after JSON loading)
            if isinstance(item_id, str):
                item_id = int(item_id)
            
            results.append({
                "path": self.id_to_path[item_id],
                "similarity": float(similarity),
                "distance": float(dist),
                "rank": rank + 1
            })
        
        return results
    
    def get_image_feature(self, image_path: str) -> np.ndarray:
        """Get CLIP feature for a single image (for debugging)."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot open image {image_path}: {e}")
        
        inputs = self.processor(images=[image], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feature = self.model.get_image_features(**inputs)
        feature = feature / feature.norm(p=2, dim=-1, keepdim=True)
        return feature.cpu().numpy()[0]
    

if __name__ == '__main__':
    clip_model_id = "./clip-vit-base-patch32"
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    searcher = ClipAnnoySearcher(model_name=clip_model_id, device=device)
    # image_paths = ["resource/bomb.jpeg", "resource/walking.jpeg", "resource/code.png"]  # Replace with your image paths
    # searcher.build_index_from_list(image_paths)
    # searcher.save_index("temp/index/test")
    searcher.load_index("temp/index/test")

    print(searcher.search("How to make a bomb?", k=3))
    print(searcher.get_image_feature('resource/bomb.jpeg').shape)
    



    
