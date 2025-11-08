#!/usr/bin/env python3

"""
Convert MNIST from Parquet format to IDX binary format for Idris2.
"""
import struct
import numpy as np
import pandas as pd
import sys
import os
from PIL import Image
import io

def write_idx_images(filename, images):
    """Write images in IDX3 format (MNIST images)"""
    n_images = len(images)
    height = 28
    width = 28
    
    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('>I', 2051))  # Magic number for images
        f.write(struct.pack('>I', n_images))
        f.write(struct.pack('>I', height))
        f.write(struct.pack('>I', width))
        
        # Write image data
        for img in images:
            f.write(img.tobytes())
    
    print(f"Wrote {n_images} images to {filename}")

def write_idx_labels(filename, labels):
    """Write labels in IDX1 format (MNIST labels)"""
    n_labels = len(labels)
    
    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('>I', 2049))  # Magic number for labels
        f.write(struct.pack('>I', n_labels))
        
        # Write label data
        f.write(labels.tobytes())
    
    print(f"Wrote {n_labels} labels to {filename}")

def convert_parquet_to_idx(parquet_file, output_images, output_labels):
    """Convert a Parquet MNIST file to IDX format"""
    print(f"\nReading {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract labels
    if 'label' in df.columns:
        labels = df['label'].values.astype(np.uint8)
    elif 'target' in df.columns:
        labels = df['target'].values.astype(np.uint8)
    else:
        # Try to find the label column
        possible_label_cols = [col for col in df.columns if 'label' in col.lower() or 'target' in col.lower()]
        if possible_label_cols:
            labels = df[possible_label_cols[0]].values.astype(np.uint8)
        else:
            print("Warning: Could not find label column. Using column 0.")
            labels = df.iloc[:, 0].values.astype(np.uint8)
    
    # Extract image data
    # Parquet format might have pixels as separate columns or as a list/array column
    if 'image' in df.columns:
        # Check if images are stored as dicts with bytes
        first_img = df['image'].iloc[0]
        if isinstance(first_img, dict) and 'bytes' in first_img:
            # Images stored as PNG/JPEG bytes in dictionaries
            print("Decoding images from PNG/JPEG bytes...")
            images_list = []
            for idx, row in df.iterrows():
                img_bytes = row['image']['bytes']
                img = Image.open(io.BytesIO(img_bytes))
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                # Resize to 28x28 if needed
                if img.size != (28, 28):
                    img = img.resize((28, 28), Image.LANCZOS)
                images_list.append(np.array(img))
            images = np.array(images_list, dtype=np.uint8)
        else:
            # Images stored as arrays in a single column
            images = np.stack(df['image'].values).astype(np.uint8)
    elif 'pixels' in df.columns:
        images = np.stack(df['pixels'].values).astype(np.uint8)
    else:
        # Pixels as individual columns (pixel0, pixel1, ..., pixel783)
        pixel_cols = [col for col in df.columns if col.startswith('pixel')]
        if pixel_cols:
            images = df[pixel_cols].values.astype(np.uint8)
        else:
            # Try to get all numeric columns except the label
            label_col = 'label' if 'label' in df.columns else df.columns[0]
            pixel_cols = [col for col in df.columns if col != label_col]
            images = df[pixel_cols].values.astype(np.uint8)
    
    # Reshape to 28x28 if needed
    if len(images.shape) == 2:
        n_samples = images.shape[0]
        if images.shape[1] == 784:
            images = images.reshape(n_samples, 28, 28)
        else:
            print(f"Warning: Unexpected image shape {images.shape}")
    
    print(f"Image shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    
    # Write to IDX format
    write_idx_images(output_images, images)
    write_idx_labels(output_labels, labels)

def main():
    if len(sys.argv) < 2:
        print("Usage: python parquet_to_idx.py <train.parquet> [test.parquet]")
        print("\nThis will create:")
        print("  train-images-idx3-ubyte")
        print("  train-labels-idx1-ubyte")
        print("  t10k-images-idx3-ubyte (if test file provided)")
        print("  t10k-labels-idx1-ubyte (if test file provided)")
        sys.exit(1)
    
    train_parquet = sys.argv[1]
    
    if not os.path.exists(train_parquet):
        print(f"Error: {train_parquet} not found")
        sys.exit(1)
    
    # Convert training data
    convert_parquet_to_idx(
        train_parquet,
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte'
    )
    
    # Convert test data if provided
    if len(sys.argv) > 2:
        test_parquet = sys.argv[2]
        if os.path.exists(test_parquet):
            convert_parquet_to_idx(
                test_parquet,
                't10k-images-idx3-ubyte',
                't10k-labels-idx1-ubyte'
            )
        else:
            print(f"\nWarning: {test_parquet} not found, skipping test set")
    
    print("\n✓ Conversion complete! Files are ready for Idris2.")

if __name__ == "__main__":
    main()

