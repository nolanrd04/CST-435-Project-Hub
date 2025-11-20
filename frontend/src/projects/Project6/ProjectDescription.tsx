import React from 'react';
import { AiOutlineFileText, AiOutlineInfoCircle } from 'react-icons/ai';

function ProjectDescription() {
  return (
    <div style={{ padding: '20px' }}>
      <h2
        style={{
          marginBottom: '30px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}
      >
        <span
          style={{
            background: 'linear-gradient(135deg, #667eea, #764ba2)',
            color: 'white',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '20px',
          }}
        >
          <AiOutlineFileText size={24} />
        </span>
        Project Description
      </h2>

      {/* Overview */}
      <div
        style={{
          backgroundColor: '#f8f9ff',
          border: '2px solid #667eea',
          borderRadius: '12px',
          padding: '25px',
          marginBottom: '20px',
        }}
      >
        <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Project Overview</h3>
        <div style={{ lineHeight: '1.8', color: '#333' }}>
          <p>
            This project implements a <strong>Generative Adversarial Network (GAN)</strong> using
            PyTorch to generate realistic fruit images. GANs are a type of neural network
            architecture that consists of two networks competing against each other:
          </p>
          <div style={{ marginTop: '15px', marginBottom: '15px' }}>
            <div style={{ padding: '10px', marginBottom: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px' }}>
              <strong>Generator:</strong> Creates fake images from random noise
            </div>
            <div style={{ padding: '10px', backgroundColor: '#e0e7ff', borderRadius: '6px' }}>
              <strong>Discriminator:</strong> Tries to distinguish real images from fake ones
            </div>
          </div>
          <p>
            Through this adversarial training process, the generator becomes increasingly skilled
            at creating realistic fruit images that can fool the discriminator.
          </p>
        </div>
      </div>

      {/* Dataset Information */}
      <div
        style={{
          backgroundColor: '#f0fdf4',
          border: '2px solid #22c55e',
          borderRadius: '12px',
          padding: '25px',
          marginBottom: '20px',
        }}
      >
        <h3 style={{ margin: '0 0 15px 0', color: '#22c55e' }}>Dataset: Google Quick, Draw!</h3>
        <div style={{ lineHeight: '1.8', color: '#333' }}>
          <p>
            The training data for this GAN comes from{' '}
            <strong>Google's Quick, Draw! dataset</strong>, a collection of over 50 million drawings
            across 345 categories, contributed by real people from around the world.
          </p>

          <div
            style={{
              backgroundColor: 'white',
              border: '1px solid #86efac',
              borderRadius: '8px',
              padding: '15px',
              marginTop: '15px',
              marginBottom: '15px',
            }}
          >
            <h4 style={{ margin: '0 0 10px 0', color: '#22c55e', fontSize: '16px' }}>
              What is Quick, Draw!?
            </h4>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              Quick, Draw! is an online game where players are given 20 seconds to sketch a specific
              object (like a fruit). The drawings are collected and used to train machine learning
              models to recognize hand-drawn doodles.
            </p>
          </div>

          <p>
            <strong>For this project, we use 7 fruit categories:</strong>
          </p>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
              gap: '8px',
              marginTop: '10px',
            }}
          >
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px', textAlign: 'center', fontSize: '14px' }}>
              Apple
            </div>
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px', textAlign: 'center', fontSize: '14px' }}>
              Banana
            </div>
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px', textAlign: 'center', fontSize: '14px' }}>
              Blackberry
            </div>
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px', textAlign: 'center', fontSize: '14px' }}>
              Grape
            </div>
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px', textAlign: 'center', fontSize: '14px' }}>
              Pear
            </div>
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px', textAlign: 'center', fontSize: '14px' }}>
              Strawberry
            </div>
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px', textAlign: 'center', fontSize: '14px' }}>
              Watermelon
            </div>
          </div>

          <div
            style={{
              marginTop: '15px',
              padding: '15px',
              backgroundColor: '#dcfce7',
              borderRadius: '8px',
              fontSize: '14px',
            }}
          >
            <strong>Why use hand-drawn data?</strong>
            <p style={{ margin: '5px 0 0 0' }}>
              Hand-drawn sketches from real people provide a unique artistic style that differs from
              photographs. The GAN learns to replicate this sketch-like quality, creating generated
              images that maintain the human-drawn aesthetic. This makes the dataset particularly
              interesting for demonstrating how GANs can learn and reproduce artistic styles.
            </p>
          </div>

          <p style={{ marginTop: '15px', fontSize: '13px', color: '#666', fontStyle: 'italic' }}>
            Dataset source:{' '}
            <a
              href="https://quickdraw.withgoogle.com/data"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: '#22c55e', textDecoration: 'underline' }}
            >
              Google Quick, Draw! Data
            </a>
          </p>
        </div>
      </div>

      {/* How It Works */}
      <div
        style={{
          backgroundColor: '#f8f9ff',
          border: '2px solid #667eea',
          borderRadius: '12px',
          padding: '25px',
          marginBottom: '20px',
        }}
      >
        <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>How It Works</h3>
        <div style={{ lineHeight: '1.8', color: '#333' }}>
          <div style={{ marginBottom: '20px' }}>
            <strong style={{ color: '#667eea' }}>1. Generator Network (Hybrid Architecture):</strong>
            <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Takes a 100-dimensional random noise vector as input
              </div>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Uses fully connected layers to expand the noise
              </div>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Applies transpose convolutions to progressively upsample
              </div>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Outputs a 64×64 grayscale fruit image
              </div>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Uses BatchNorm and LeakyReLU for stable training
              </div>
            </div>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <strong style={{ color: '#667eea' }}>2. Discriminator Network (CNN):</strong>
            <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Takes a 64×64 image as input (real or generated)
              </div>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Uses convolutional layers to downsample and extract features
              </div>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Outputs a single probability: real (1) or fake (0)
              </div>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Uses Dropout to prevent overfitting
              </div>
            </div>
          </div>

          <div>
            <strong style={{ color: '#667eea' }}>3. Training Process:</strong>
            <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Train discriminator to recognize real vs fake images
              </div>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Train generator to fool the discriminator
              </div>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Iterate this process for many epochs
              </div>
              <div style={{ padding: '8px', backgroundColor: '#e0e7ff', borderRadius: '6px', fontSize: '14px' }}>
                Generator improves over time at creating realistic images
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline */}
      <div
        style={{
          backgroundColor: '#f8f9ff',
          border: '2px solid #667eea',
          borderRadius: '12px',
          padding: '25px',
          marginBottom: '20px',
        }}
      >
        <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Project Pipeline</h3>
        <div style={{ lineHeight: '1.8', color: '#333' }}>
          <div style={{ marginBottom: '12px' }}>
            <strong>Step 1: Data Preparation</strong>
            <div style={{ marginLeft: '20px', fontSize: '14px', color: '#666' }}>
              Convert raw NDJSON drawing data to image format, then to NPZ arrays for training
            </div>
          </div>

          <div style={{ marginBottom: '12px' }}>
            <strong>Step 2: Model Configuration</strong>
            <div style={{ marginLeft: '20px', fontSize: '14px', color: '#666' }}>
              Set training parameters (epochs, batch size, learning rate) and choose data version
            </div>
          </div>

          <div style={{ marginBottom: '12px' }}>
            <strong>Step 3: Training</strong>
            <div style={{ marginLeft: '20px', fontSize: '14px', color: '#666' }}>
              Train separate Generator/Discriminator pairs for each of the 7 fruits. Training
              happens automatically for all fruits in one session.
            </div>
          </div>

          <div style={{ marginBottom: '12px' }}>
            <strong>Step 4: Generation</strong>
            <div style={{ marginLeft: '20px', fontSize: '14px', color: '#666' }}>
              Use trained models to generate new fruit images from random noise
            </div>
          </div>

          <div>
            <strong>Step 5: Analysis</strong>
            <div style={{ marginLeft: '20px', fontSize: '14px', color: '#666' }}>
              Review training costs, model performance, and generated image quality
            </div>
          </div>
        </div>
      </div>

      {/* Architecture Details */}
      <div
        style={{
          backgroundColor: '#f8f9ff',
          border: '2px solid #667eea',
          borderRadius: '12px',
          padding: '25px',
          marginBottom: '20px',
        }}
      >
        <h3 style={{ margin: '0 0 15px 0', color: '#667eea' }}>Technical Architecture</h3>
        <div style={{ lineHeight: '1.8', color: '#333' }}>
          <p>
            This implementation uses a <strong>DCGAN (Deep Convolutional GAN)</strong> architecture:
          </p>

          <div style={{ marginTop: '15px' }}>
            <strong>Generator Flow:</strong>
            <div
              style={{
                fontFamily: 'monospace',
                backgroundColor: '#f0f4ff',
                padding: '12px',
                borderRadius: '6px',
                marginTop: '8px',
                fontSize: '13px',
              }}
            >
              (100D noise) → FC Layer → (128, 16, 16)
              <br />
              → Upsample + Conv → (128, 32, 32)
              <br />
              → Conv → (64, 32, 32)
              <br />
              → Upsample + Conv → (64, 64, 64)
              <br />
              → Conv + Tanh → (1, 64, 64) image
            </div>
          </div>

          <div style={{ marginTop: '15px' }}>
            <strong>Discriminator Flow:</strong>
            <div
              style={{
                fontFamily: 'monospace',
                backgroundColor: '#f0f4ff',
                padding: '12px',
                borderRadius: '6px',
                marginTop: '8px',
                fontSize: '13px',
              }}
            >
              (1, 64, 64) image → Conv → (32, 32, 32)
              <br />
              → Conv → (64, 16, 16)
              <br />
              → Conv → (128, 8, 8)
              <br />
              → Flatten + FC + Sigmoid → probability [0, 1]
            </div>
          </div>
        </div>
      </div>

      {/* Key Features */}
      <div
        style={{
          backgroundColor: '#f0fff4',
          border: '2px solid #48bb78',
          borderRadius: '12px',
          padding: '25px',
          marginBottom: '20px',
        }}
      >
        <h3
          style={{
            margin: '0 0 15px 0',
            color: '#48bb78',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <AiOutlineInfoCircle size={24} />
          Key Features
        </h3>
        <div style={{ lineHeight: '1.8', color: '#333' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{ padding: '12px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px' }}>
              <strong>Multi-Fruit Training:</strong> Separate models for 7 different fruits trained
              in one session
            </div>
            <div style={{ padding: '12px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px' }}>
              <strong>Cost Tracking:</strong> Comprehensive training cost analysis including
              compute, memory, and storage
            </div>
            <div style={{ padding: '12px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px' }}>
              <strong>Flexible Data Versions:</strong> Create new datasets or reuse existing ones
            </div>
            <div style={{ padding: '12px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px' }}>
              <strong>Configurable Parameters:</strong> Customize image resolution, training
              epochs, batch size, and learning rate
            </div>
            <div style={{ padding: '12px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px' }}>
              <strong>Progress Monitoring:</strong> Track training progress with epoch-by-epoch
              checkpoints
            </div>
            <div style={{ padding: '12px', backgroundColor: 'white', border: '1px solid #86efac', borderRadius: '6px' }}>
              <strong>Model Persistence:</strong> All models, configurations, and training
              histories saved for analysis
            </div>
          </div>
        </div>
      </div>

      {/* Thought Process */}
      <div
        style={{
          backgroundColor: '#fff5f0',
          border: '2px solid #ff7b29',
          borderRadius: '12px',
          padding: '25px',
        }}
      >
        <h3 style={{ margin: '0 0 15px 0', color: '#ff7b29' }}>Design Thought Process</h3>
        <div style={{ lineHeight: '1.8', color: '#333' }}>
          <p>
            <strong>Why GANs for Image Generation?</strong>
          </p>
          <p style={{ marginLeft: '20px', fontSize: '14px', color: '#666' }}>
            GANs excel at learning the underlying distribution of training data and generating new
            samples that appear authentic. Unlike traditional generative models, GANs don't require
            explicit probability distributions, making them ideal for complex image generation
            tasks.
          </p>

          <p style={{ marginTop: '15px' }}>
            <strong>Why Separate Models Per Fruit?</strong>
          </p>
          <p style={{ marginLeft: '20px', fontSize: '14px', color: '#666' }}>
            Each fruit has unique visual characteristics. Training separate models allows each
            generator to specialize in the specific features of its target fruit, resulting in
            higher quality outputs compared to a single multi-class model.
          </p>

          <p style={{ marginTop: '15px' }}>
            <strong>Architecture Choices:</strong>
          </p>
          <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #fed7aa', borderRadius: '6px', fontSize: '14px', color: '#666' }}>
              <strong>BatchNorm:</strong> Stabilizes training and allows higher learning rates
            </div>
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #fed7aa', borderRadius: '6px', fontSize: '14px', color: '#666' }}>
              <strong>LeakyReLU:</strong> Prevents dying neurons better than standard ReLU
            </div>
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #fed7aa', borderRadius: '6px', fontSize: '14px', color: '#666' }}>
              <strong>Dropout:</strong> Prevents discriminator from overfitting too quickly
            </div>
            <div style={{ padding: '8px', backgroundColor: 'white', border: '1px solid #fed7aa', borderRadius: '6px', fontSize: '14px', color: '#666' }}>
              <strong>Tanh Output:</strong> Maps generated images to [-1, 1] range matching normalized data
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ProjectDescription;
