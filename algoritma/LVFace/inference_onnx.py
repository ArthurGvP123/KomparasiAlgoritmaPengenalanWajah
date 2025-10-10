# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import requests
import cv2
import numpy as np
import onnxruntime
from typing import Optional, Tuple, Union

class LVFaceONNXInferencer:
    """LVFace Inference Class using ONNX Runtime"""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize the LVFace ONNX inferencer
        
        Args:
            model_path (str): Path to the ONNX model file
            use_gpu (bool): Whether to use GPU acceleration (requires onnxruntime-gpu)
        """
        # Select execution provider
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        # Initialize ONNX Runtime session
        self.ort_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers
        )
        
        # Get input and output names
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        # Input image size
        self.input_size = (112, 112)

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for LVFace inference
        
        Args:
            img (np.ndarray): Input image in BGR format (from cv2.imread)
            
        Returns:
            np.ndarray: Preprocessed image tensor
        """
        # Resize image to input size
        img_resized = cv2.resize(img, self.input_size)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Transpose to (C, H, W)
        img_transposed = np.transpose(img_rgb, (2, 0, 1))
        
        # Normalize to [-1, 1]
        img_normalized = ((img_transposed / 255.0) - 0.5) / 0.5
        
        # Convert to float32 and add batch dimension
        img_tensor = img_normalized.astype(np.float32)[np.newaxis, ...]
        
        return img_tensor

    def infer_from_image(self, img_path: str) -> np.ndarray:
        """
        Extract feature from a local image file
        
        Args:
            img_path (str): Path to the local image file
            
        Returns:
            np.ndarray: Extracted feature embedding
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image from {img_path}")
            
        # Preprocess image
        img_tensor = self._preprocess_image(img)
        
        # Run inference
        output = self.ort_session.run(
            [self.output_name],
            {self.input_name: img_tensor}
        )
        
        return output[0]

    def infer_from_url(self, img_url: str) -> np.ndarray:
        """
        Extract feature from an image URL
        
        Args:
            img_url (str): URL of the image
            
        Returns:
            np.ndarray: Extracted feature embedding
        """
        try:
            # Download image from URL
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Convert to numpy array
            np_arr = np.frombuffer(response.content, np.uint8)
            
            # Decode image
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from URL content")
                
            # Preprocess and infer
            img_tensor = self._preprocess_image(img)
            output = self.ort_session.run(
                [self.output_name],
                {self.input_name: img_tensor}
            )
            
            return output[0]
            
        except Exception as e:
            raise RuntimeError(f"Error processing image from URL: {str(e)}")

    @staticmethod
    def calculate_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two features
        
        Args:
            feat1 (np.ndarray): First feature embedding
            feat2 (np.ndarray): Second feature embedding
            
        Returns:
            float: Cosine similarity score (range: [-1, 1])
        """
        # Flatten features
        feat1_flat = np.ravel(feat1)
        feat2_flat = np.ravel(feat2)
        
        # Calculate cosine similarity
        dot_product = np.dot(feat1_flat, feat2_flat)
        norm1 = np.linalg.norm(feat1_flat)
        norm2 = np.linalg.norm(feat2_flat)
        
        return dot_product / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0


if __name__ == "__main__":
    # Example usage
    model_path = "./LVFace-T_Glint360K.onnx"  # Update with your model path
    inferencer = LVFaceONNXInferencer(model_path, use_gpu=True)
    
    # Example 1: Inference from local image
    try:
        img_path = "1.jpg"  # Update with your image path
        embedding = inferencer.infer_from_image(img_path)
        print(f"Extracted feature shape: {embedding.shape}")
    except Exception as e:
        print(f"Error in image inference: {e}")
    
    # Example 2: Inference from URL
    try:
        img_url = "url"
        embedding_url = inferencer.infer_from_url(img_url)
        print(f"Extracted feature from URL shape: {embedding_url.shape}")
    except Exception as e:
        print(f"Error in URL inference: {e}")
    
    # Example 3: Calculate similarity between two images
    try:
        feat1 = inferencer.infer_from_image(img_path)
        feat2 = inferencer.infer_from_url(img_url)
        similarity = inferencer.calculate_similarity(feat1, feat2)
        print(f"Cosine similarity: {similarity:.6f}")
    except Exception as e:
        print(f"Error in similarity calculation: {e}")