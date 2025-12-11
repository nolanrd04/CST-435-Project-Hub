import React from 'react';
import { AiOutlineDatabase, AiOutlineCode, AiOutlineFileText } from 'react-icons/ai';
import trainingHistoryImg from './training_history.png';

function AssignmentDetails() {
  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Data Section */}
      <div
        style={{
          marginBottom: '40px',
          background: 'white',
          borderRadius: '12px',
          padding: '30px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}
      >
        <h2
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '25px',
            color: '#667eea',
            fontSize: '24px',
            fontWeight: 'bold',
          }}
        >
          <AiOutlineDatabase size={28} />
          Data
        </h2>

        {/* Descriptive Analysis */}
        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ color: '#333', fontSize: '18px', marginBottom: '15px', fontWeight: '600' }}>
            Descriptive Analysis of the Data
          </h3>
          <div style={{ color: '#555', lineHeight: '1.8', fontSize: '15px' }}>
            <div style={{ marginBottom: '12px' }}>
              The dataset consists of <strong>2,500 fruit images</strong> collected for training a U-Net image colorization model. The dataset is perfectly balanced with <strong>500 images each</strong> of five fruit categories: orange, strawberry, banana, pineapple, and blackberry.
            </div>
            <div style={{ marginBottom: '12px' }}>
              <strong>Dataset Characteristics:</strong>
            </div>
            <div style={{ paddingLeft: '20px', marginBottom: '12px' }}>
              <div>• Image format: PNG and JPG</div>
              <div>• Resolution: Standardized to 128×128 pixels</div>
              <div>• Content diversity: Multi-fruit images (bowls of strawberries, bunches of bananas), single fruits, and various quantities</div>
              <div>• Background variations: Studio/clean backgrounds (white or transparent), natural environments (kitchens, markets, orchards), and wild/outdoor settings (trees, bushes, farms)</div>
            </div>
            <div style={{ marginBottom: '12px' }}>
              This diverse composition ensures the colorization model learns to handle various lighting conditions, backgrounds, and fruit presentations that users might encounter in real-world applications. The images are split into training (70%), validation (15%), and test (15%) sets using stratified sampling to ensure equal representation across all fruit classes.
            </div>
          </div>
        </div>

        {/* Normalization/Standardization */}
        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ color: '#333', fontSize: '18px', marginBottom: '15px', fontWeight: '600' }}>
            Normalization and Standardization
          </h3>
          <div style={{ color: '#555', lineHeight: '1.8', fontSize: '15px' }}>
            <div style={{ marginBottom: '12px' }}>
              <strong>The data requires both normalization and standardization.</strong>
            </div>
            <div style={{ marginBottom: '12px' }}>
              The preprocessing pipeline applies a two-step transformation:
            </div>
            <div
              style={{
                background: '#f5f5f5',
                padding: '15px',
                borderRadius: '8px',
                fontFamily: 'monospace',
                fontSize: '14px',
                marginBottom: '12px',
              }}
            >
              <div>transforms.ToTensor()              # [0, 255] → [0, 1]</div>
              <div>transforms.Normalize([0.5], [0.5]) # [0, 1] → [-1, 1]</div>
            </div>
            <div style={{ marginBottom: '12px' }}>
              <strong>Why [-1, 1] normalization is critical:</strong>
            </div>
            <div style={{ paddingLeft: '20px', marginBottom: '12px' }}>
              <div>• <strong>Neural Network Stability:</strong> Prevents gradient explosion and vanishing gradients during training</div>
              <div>• <strong>Activation Function Efficiency:</strong> Tanh and sigmoid activations work optimally in the [-1, 1] range</div>
              <div>• <strong>Color Reconstruction Accuracy:</strong> Symmetric range preserves color relationships and gradients</div>
              <div>• <strong>Training Speed:</strong> Normalized inputs enable faster convergence and more stable learning</div>
              <div>• <strong>Model Generalization:</strong> Consistent input ranges improve model performance on unseen data</div>
            </div>
            <div style={{ marginBottom: '12px' }}>
              This standardization approach centers the data around zero with a standard deviation of 1, which is mathematically optimal for deep neural networks using modern activation functions.
            </div>
          </div>
        </div>

        {/* Data Cleaning */}
        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ color: '#333', fontSize: '18px', marginBottom: '15px', fontWeight: '600' }}>
            Data Cleaning and Missing Values
          </h3>
          <div style={{ color: '#555', lineHeight: '1.8', fontSize: '15px' }}>
            <div style={{ marginBottom: '12px' }}>
              The data collection system implements a comprehensive validation pipeline to ensure high-quality training data:
            </div>
            <div style={{ paddingLeft: '20px', marginBottom: '12px' }}>
              <div style={{ marginBottom: '8px' }}>
                <strong>1. URL Validation:</strong> Invalid, empty, or malformed URLs are filtered out before download attempts.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>2. Network Connectivity Validation:</strong> Unreachable images due to server errors, timeouts, or network failures are automatically skipped with retry logic.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>3. Content Size Validation:</strong> Files smaller than 1000 bytes are rejected as they indicate incomplete downloads or broken images.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>4. Image Format Validation:</strong> All images undergo PIL verification checks. Corrupted or unreadable files are filtered out.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>5. Format Standardization:</strong> Images with transparency (RGBA, LA modes) are converted to RGB by compositing onto a white background, ensuring consistent 3-channel output.
              </div>
            </div>
            <div style={{ marginBottom: '12px' }}>
              <strong>Types of "Missing Values" Handled:</strong>
            </div>
            <div style={{ paddingLeft: '20px', marginBottom: '12px' }}>
              <div>• Invalid URLs (empty, malformed, or non-HTTP)</div>
              <div>• Network failures (unreachable servers, timeouts)</div>
              <div>• Corrupted files (failed PIL verification)</div>
              <div>• Insufficient data (files smaller than 1000 bytes)</div>
              <div>• Format inconsistencies (non-RGB images converted to standard RGB)</div>
              <div>• Content validation (URLs filtered based on relevance to fruit images)</div>
            </div>
            <div style={{ marginBottom: '12px' }}>
              The system maintains detailed statistics tracking total attempts, successful downloads, errors, filtered images, and duplicates removed, ensuring complete transparency in the data cleaning process.
            </div>
          </div>
        </div>

        {/* Outlier Handling */}
        <div style={{ marginBottom: '0' }}>
          <h3 style={{ color: '#333', fontSize: '18px', marginBottom: '15px', fontWeight: '600' }}>
            Outlier Handling
          </h3>
          <div style={{ color: '#555', lineHeight: '1.8', fontSize: '15px' }}>
            <div style={{ marginBottom: '12px' }}>
              The dataset collection system implements multi-level outlier detection to ensure training data quality and consistency. Based on typical collection runs, the system achieves a <strong>40% overall rejection rate</strong> ensuring high-quality training data.
            </div>

            <div style={{ marginBottom: '12px' }}>
              <strong>Dimensional Outliers:</strong>
            </div>
            <div style={{ paddingLeft: '20px', marginBottom: '16px' }}>
              <div style={{ marginBottom: '8px' }}>
                <strong>1. Size-Based Filtering (~15% rejection rate):</strong> Images with minimum dimension below 64 pixels are rejected as they lack sufficient resolution for meaningful colorization.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>2. Aspect Ratio Constraints (~10% rejection rate):</strong> Images with aspect ratios outside [0.2, 5.0] are filtered out, removing extremely wide or tall images that don't represent typical fruit photographs.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>3. File Size Limits:</strong> Images exceeding 10MB are rejected to prevent memory issues and likely indicate unnecessarily high resolution or encoding artifacts.
              </div>
            </div>

            <div style={{ marginBottom: '12px' }}>
              <strong>Content-Based Outlier Detection (~7% rejection rate):</strong>
            </div>
            <div style={{ paddingLeft: '20px', marginBottom: '16px' }}>
              <div style={{ marginBottom: '8px' }}>
                <strong>Semantic URL Filtering:</strong> URLs containing non-fruit keywords (sunset, beach, sky, ocean, sand, landscape) are filtered out. This originated from beach images slipping through data-scraping but now provides better overall data filtering.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>Image Format Validation:</strong> Only URLs pointing to actual image files (.jpg, .jpeg, .png, .webp, .gif) are accepted.
              </div>
            </div>

            <div style={{ marginBottom: '12px' }}>
              <strong>Advanced Duplicate Detection (~8% rejection rate):</strong>
            </div>
            <div style={{ paddingLeft: '20px', marginBottom: '16px' }}>
              <div style={{ marginBottom: '8px' }}>
                The system uses three perceptual hashing algorithms (pHash, dHash, wHash) to detect visually similar or identical images. This multi-hash approach ensures 500 unique images per fruit category and prevents model overfitting to repeated samples.
              </div>
            </div>

            <div style={{ marginBottom: '12px' }}>
              <strong>Search Quality Control:</strong>
            </div>
            <div style={{ paddingLeft: '20px', marginBottom: '12px' }}>
              <div>• Search terms are randomized to prevent result clustering</div>
              <div>• Term variations ("high quality", "macro photography", "studio shot") are added for diversity</div>
              <div>• Hybrid selection combines top relevant results with randomized selections for balance</div>
            </div>

            <div>
              This comprehensive outlier detection ensures the U-Net colorizer receives clean, consistent, and diverse training data optimized for fruit image colorization tasks.
            </div>
          </div>
        </div>
      </div>

      {/* Model Section */}
      <div
        style={{
          marginBottom: '40px',
          background: 'white',
          borderRadius: '12px',
          padding: '30px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}
      >
        <h2
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '25px',
            color: '#667eea',
            fontSize: '24px',
            fontWeight: 'bold',
          }}
        >
          <AiOutlineCode size={28} />
          Model
        </h2>

        {/* Theoretical Foundation */}
        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ color: '#333', fontSize: '18px', marginBottom: '15px', fontWeight: '600' }}>
            Theoretical Foundation
          </h3>
          <div style={{ color: '#555', lineHeight: '1.8', fontSize: '15px' }}>
            <div style={{ marginBottom: '16px' }}>
              The model employs a <strong>U-Net architecture</strong>, an encoder-decoder convolutional neural network with skip connections, specifically designed for image-to-image translation tasks.
            </div>
            <div style={{ marginBottom: '16px' }}>
              <strong>Mathematical Formulation:</strong>
            </div>
            <div
              style={{
                background: '#f5f5f5',
                padding: '20px',
                borderRadius: '8px',
                fontFamily: 'monospace',
                fontSize: '14px',
                marginBottom: '16px',
                overflowX: 'auto',
              }}
            >
              <div style={{ marginBottom: '12px' }}>
                <strong>Input:</strong> Grayscale image <em>X</em> ∈ ℝ<sup>H×W×1</sup> where H = W = 128
              </div>
              <div style={{ marginBottom: '12px' }}>
                <strong>Output:</strong> RGB image <em>Y</em> ∈ ℝ<sup>H×W×3</sup>
              </div>
              <div style={{ marginBottom: '16px' }}>
                <strong>Objective:</strong> Learn mapping <em>f</em>: ℝ<sup>H×W×1</sup> → ℝ<sup>H×W×3</sup> such that <em>f(X) ≈ Y</em>
              </div>

              <div style={{ marginBottom: '12px' }}>
                <strong>Encoder (Downsampling Path):</strong>
              </div>
              <div style={{ paddingLeft: '15px', marginBottom: '16px' }}>
                <div><em>e</em><sub>1</sub> = ReLU(BN(Conv<sub>3×3</sub>(X, 64)))</div>
                <div><em>p</em><sub>1</sub> = MaxPool<sub>2×2</sub>(<em>e</em><sub>1</sub>)</div>
                <div><em>e</em><sub>2</sub> = ReLU(BN(Conv<sub>3×3</sub>(<em>p</em><sub>1</sub>, 128)))</div>
                <div><em>p</em><sub>2</sub> = MaxPool<sub>2×2</sub>(<em>e</em><sub>2</sub>)</div>
                <div><em>e</em><sub>3</sub> = ReLU(BN(Conv<sub>3×3</sub>(<em>p</em><sub>2</sub>, 256)))</div>
                <div><em>p</em><sub>3</sub> = MaxPool<sub>2×2</sub>(<em>e</em><sub>3</sub>)</div>
                <div><em>e</em><sub>4</sub> = ReLU(BN(Conv<sub>3×3</sub>(<em>p</em><sub>3</sub>, 512)))</div>
                <div><em>p</em><sub>4</sub> = MaxPool<sub>2×2</sub>(<em>e</em><sub>4</sub>)</div>
              </div>

              <div style={{ marginBottom: '12px' }}>
                <strong>Bottleneck:</strong>
              </div>
              <div style={{ paddingLeft: '15px', marginBottom: '16px' }}>
                <div><em>b</em> = ReLU(BN(Conv<sub>3×3</sub>(<em>p</em><sub>4</sub>, 1024)))</div>
              </div>

              <div style={{ marginBottom: '12px' }}>
                <strong>Decoder (Upsampling Path with Skip Connections):</strong>
              </div>
              <div style={{ paddingLeft: '15px', marginBottom: '16px' }}>
                <div><em>u</em><sub>4</sub> = UpConv<sub>2×2</sub>(<em>b</em>, 512)</div>
                <div><em>d</em><sub>4</sub> = ReLU(BN(Conv<sub>3×3</sub>(Concat(<em>u</em><sub>4</sub>, <em>e</em><sub>4</sub>), 512)))</div>
                <div><em>u</em><sub>3</sub> = UpConv<sub>2×2</sub>(<em>d</em><sub>4</sub>, 256)</div>
                <div><em>d</em><sub>3</sub> = ReLU(BN(Conv<sub>3×3</sub>(Concat(<em>u</em><sub>3</sub>, <em>e</em><sub>3</sub>), 256)))</div>
                <div><em>u</em><sub>2</sub> = UpConv<sub>2×2</sub>(<em>d</em><sub>3</sub>, 128)</div>
                <div><em>d</em><sub>2</sub> = ReLU(BN(Conv<sub>3×3</sub>(Concat(<em>u</em><sub>2</sub>, <em>e</em><sub>2</sub>), 128)))</div>
                <div><em>u</em><sub>1</sub> = UpConv<sub>2×2</sub>(<em>d</em><sub>2</sub>, 64)</div>
                <div><em>d</em><sub>1</sub> = ReLU(BN(Conv<sub>3×3</sub>(Concat(<em>u</em><sub>1</sub>, <em>e</em><sub>1</sub>), 64)))</div>
              </div>

              <div style={{ marginBottom: '12px' }}>
                <strong>Output Layer:</strong>
              </div>
              <div style={{ paddingLeft: '15px', marginBottom: '16px' }}>
                <div><em>Y</em> = σ(Conv<sub>1×1</sub>(<em>d</em><sub>1</sub>, 3))</div>
                <div style={{ fontSize: '13px', color: '#666', marginTop: '6px' }}>
                  where σ is the sigmoid activation function, ensuring output ∈ [0, 1]<sup>H×W×3</sup>
                </div>
              </div>

              <div style={{ marginBottom: '12px' }}>
                <strong>Loss Function:</strong>
              </div>
              <div style={{ paddingLeft: '15px' }}>
                <div>ℒ(<em>Y</em>, <em>Ŷ</em>) = MSE(<em>Y</em>, <em>Ŷ</em>) = (1/HWC) Σ<sub>i,j,c</sub> (<em>Y</em><sub>i,j,c</sub> - <em>Ŷ</em><sub>i,j,c</sub>)<sup>2</sup></div>
                <div style={{ fontSize: '13px', color: '#666', marginTop: '6px' }}>
                  Alternative: L1 loss for robustness to outliers
                </div>
              </div>
            </div>
            <div style={{ marginBottom: '12px' }}>
              <strong>Key Components:</strong>
            </div>
            <div style={{ paddingLeft: '20px', marginBottom: '12px' }}>
              <div>• <strong>Conv<sub>3×3</sub>:</strong> 3×3 convolution with padding=1 to preserve spatial dimensions</div>
              <div>• <strong>BN:</strong> Batch Normalization for training stability and faster convergence</div>
              <div>• <strong>ReLU:</strong> Rectified Linear Unit activation, <em>f(x) = max(0, x)</em></div>
              <div>• <strong>MaxPool<sub>2×2</sub>:</strong> 2×2 max pooling with stride=2 for downsampling</div>
              <div>• <strong>UpConv<sub>2×2</sub>:</strong> 2×2 transposed convolution with stride=2 for upsampling</div>
              <div>• <strong>Concat:</strong> Channel-wise concatenation of encoder and decoder features (skip connections)</div>
            </div>
            <div>
              The skip connections are critical as they preserve high-resolution spatial information from the encoder, enabling precise localization in the decoder for accurate pixel-level colorization.
            </div>
          </div>
        </div>

        {/* Data Processing */}
        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ color: '#333', fontSize: '18px', marginBottom: '15px', fontWeight: '600' }}>
            Data Processing Pipeline
          </h3>
          <div style={{ color: '#555', lineHeight: '1.8', fontSize: '15px' }}>
            <div style={{ marginBottom: '12px' }}>
              The data undergoes a multi-stage preprocessing pipeline before being fed to the model:
            </div>
            <div style={{ paddingLeft: '20px', marginBottom: '12px' }}>
              <div style={{ marginBottom: '8px' }}>
                <strong>1. Loading and Validation:</strong> Color images are loaded from disk and validated for integrity.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>2. Resizing:</strong> Images are resized to 128×128 pixels using LANCZOS interpolation to maintain quality while standardizing dimensions.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>3. Grayscale Conversion:</strong> RGB images are converted to grayscale using the PIL convert('L') method which applies the luminance-preserving formula.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>4. Normalization and Standardization:</strong> Images are transformed to tensors and normalized:
                <div
                  style={{
                    background: '#f5f5f5',
                    padding: '10px',
                    borderRadius: '6px',
                    fontFamily: 'monospace',
                    fontSize: '14px',
                    marginTop: '8px',
                  }}
                >
                  <div>transforms.ToTensor()              # [0, 255] → [0, 1]</div>
                  <div>transforms.Normalize([0.5], [0.5]) # [0, 1] → [-1, 1]</div>
                </div>
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>5. Tensor Conversion:</strong> Images are converted to PyTorch tensors with shape (B, C, H, W) where B=batch size, C=channels (1 for input, 3 for target), H=W=128.
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>6. Data Augmentation (Training Only):</strong>
                <div style={{ paddingLeft: '20px', marginTop: '6px' }}>
                  <div>• Random horizontal flip (50% probability)</div>
                  <div>• Random rotation (±15 degrees)</div>
                  <div>• Brightness adjustment (±20%)</div>
                  <div>Note: Augmentations are applied identically to both input and target to maintain correspondence</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Training History */}
        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ color: '#333', fontSize: '18px', marginBottom: '15px', fontWeight: '600' }}>
            Training History
          </h3>
          <div style={{ color: '#555', lineHeight: '1.8', fontSize: '15px', marginBottom: '15px' }}>
            The training history visualizes the model's learning progress over epochs, showing both training and validation loss curves alongside the learning rate schedule.
          </div>
          <div
            style={{
              textAlign: 'center',
              background: '#f9f9f9',
              padding: '20px',
              borderRadius: '8px',
            }}
          >
            <img
              src={trainingHistoryImg}
              alt="Training History"
              style={{
                maxWidth: '100%',
                height: 'auto',
                borderRadius: '8px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              }}
            />
            <div style={{ marginTop: '12px', fontSize: '14px', color: '#666' }}>
              Training and validation loss curves with learning rate schedule
            </div>
          </div>
        </div>

        {/* Model Tuning Parameters */}
        <div style={{ marginBottom: '0' }}>
          <h3 style={{ color: '#333', fontSize: '18px', marginBottom: '15px', fontWeight: '600' }}>
            Model Tuning Parameters
          </h3>
          <div style={{ color: '#555', lineHeight: '1.8', fontSize: '15px' }}>
            <div style={{ marginBottom: '12px' }}>
              The following hyperparameters are used to optimize model performance:
            </div>

            <div style={{ marginBottom: '16px' }}>
              <strong>Core Hyperparameters:</strong>
              <div
                style={{
                  background: '#f5f5f5',
                  padding: '15px',
                  borderRadius: '8px',
                  marginTop: '8px',
                  fontFamily: 'monospace',
                  fontSize: '14px',
                }}
              >
                <div>• Image Size: 128×128 pixels</div>
                <div>• Batch Size: 16 images per batch</div>
                <div>• Learning Rate: 0.0001 (initial)</div>
                <div>• Epochs: 50 (maximum)</div>
                <div>• Optimizer: Adam (adaptive learning rate)</div>
                <div>• Loss Function: MSE or L1 (selectable)</div>
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <strong>Learning Rate Scheduler:</strong>
              <div style={{ paddingLeft: '20px', marginTop: '8px' }}>
                <div>• Type: ReduceLROnPlateau</div>
                <div>• Patience: 5 epochs without validation improvement</div>
                <div>• Reduction Factor: 0.5 (halves learning rate)</div>
                <div>• Purpose: Automatically reduces learning rate when validation loss plateaus, enabling fine-tuning and preventing overshooting local minima</div>
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <strong>Early Stopping:</strong>
              <div style={{ paddingLeft: '20px', marginTop: '8px' }}>
                <div>• Patience: 10 epochs without validation improvement</div>
                <div>• Purpose: Prevents overfitting by halting training when the model stops improving on validation data</div>
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <strong>Regularization Techniques:</strong>
              <div style={{ paddingLeft: '20px', marginTop: '8px' }}>
                <div>• Batch Normalization: Applied after each convolutional layer to normalize activations and reduce internal covariate shift</div>
                <div>• Data Augmentation: Increases training data diversity to improve generalization</div>
                <div>• Validation Split: 15% of data reserved for validation to monitor overfitting</div>
              </div>
            </div>

            <div>
              <strong>Model Checkpointing:</strong> The model with the lowest validation loss is automatically saved, ensuring the best-performing version is retained even if training continues past the optimal point.
            </div>
          </div>
        </div>
      </div>

      {/* Summary Section */}
      <div
        style={{
          marginBottom: '40px',
          background: 'white',
          borderRadius: '12px',
          padding: '30px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}
      >
        <h2
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '25px',
            color: '#667eea',
            fontSize: '24px',
            fontWeight: 'bold',
          }}
        >
          <AiOutlineFileText size={28} />
          Summary
        </h2>
        <div
          style={{
            color: '#555',
            lineHeight: '1.8',
            fontSize: '15px',
            padding: '20px',
            background: '#f9f9ff',
            borderRadius: '8px',
            border: '2px dashed #667eea',
          }}
        >
          <div style={{ fontStyle: 'italic', color: '#667eea', marginBottom: '10px' }}>
            Overall findings
          </div>
          <div style={{ color: '#888', fontSize: '14px' }}>
            After making a diffusion model which was trained on thousands of different colored images in order to color black and white images, the model was only able to color based on shape. We talked as a group about what could we done to do this, and the idea we came up with was to use two models, one to recognzie the object, and one to color it. This is extremely resources heavy, especially for making a model to color all images, so we changed the problem scope to be an image colorer for fruits. This means we could change the architecture to be a U-net model, and it works at coloring trained fruits, as well as new fruits, extremely well.
          </div>
        </div>
      </div>
    </div>
  );
}

export default AssignmentDetails;
