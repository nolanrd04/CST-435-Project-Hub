import React, { useState, useEffect } from 'react';
import {
  AiOutlineInfoCircle,
  AiOutlineWarning,
  AiOutlineCheckCircle,
  AiOutlineBulb,
  AiOutlineThunderbolt,
  AiOutlinePlayCircle,
} from 'react-icons/ai';
import {
  GiBanana,
  GiStrawberry,
  GiOrange,
  GiPineapple,
  GiRaspberry,
} from 'react-icons/gi';
import { project9API, ModelInfo } from './api.ts';

// Import showcase images
import basketBlackberries from './ProjectDescription/Colored_a_basket_of_blackberries.png';
import bunchBananas from './ProjectDescription/Colored_a_bunch_of_bananas.png';
import bunchStrawberries from './ProjectDescription/Colored_a_bunch_of_strawberries_together.png';
import slicedOrange from './ProjectDescription/colored_a_sliced_orange.png';
import slicedPineapple from './ProjectDescription/Colored_a_sliced_pineapple.png';
import untrainedApple from './ProjectDescription/Colored_an_untrained_apple.png';
import untrainedWatermelon from './ProjectDescription/Colored_an_untrained_watermelon.png';
import bwStrawberry from './ProjectDescription/Colored_black_and_white_strawberry.png';
import blackberriesRaspberries from './ProjectDescription/Colored_blackberries_along_with_untrained_raspberries.png';
import heldBlackberries from './ProjectDescription/Colored_held_blackberries.png';
import cartoonFruits from './ProjectDescription/Colored_select_cartoon_fruits.png';

function ModelInformation() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [loadingInfo, setLoadingInfo] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [activeSection, setActiveSection] = useState<'overview' | 'technical'>('overview');

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      loadModelInfo(selectedModel);
    }
  }, [selectedModel]);

  const loadModels = async () => {
    setLoading(true);
    setError('');
    try {
      const modelList = await project9API.listModels();
      setModels(modelList);
      if (modelList.length > 0) {
        setSelectedModel(modelList[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  const loadModelInfo = async (modelName: string) => {
    setLoadingInfo(true);
    setError('');
    try {
      const info = await project9API.getModelInfo(modelName);
      setModelInfo(info);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model information');
    } finally {
      setLoadingInfo(false);
    }
  };

  const trainedFruits = [
    { name: 'Bananas', icon: GiBanana, description: 'Yellow elongated fruit' },
    { name: 'Blackberries', icon: GiRaspberry, description: 'Dark purple/black compound berry' },
    { name: 'Pineapples', icon: GiPineapple, description: 'Golden body with green crown' },
    { name: 'Oranges', icon: GiOrange, description: 'Orange peel, round fruit' },
    { name: 'Strawberries', icon: GiStrawberry, description: 'Red fruit with green leaves' },
  ];

  const showcaseImages = [
    {
      src: bunchStrawberries,
      caption: 'A bunch of strawberries together',
      category: 'Trained',
    },
    {
      src: bunchBananas,
      caption: 'A bunch of bananas',
      category: 'Trained',
    },
    {
      src: basketBlackberries,
      caption: 'A basket of blackberries',
      category: 'Trained',
    },
    {
      src: slicedOrange,
      caption: 'A sliced orange',
      category: 'Trained',
    },
    {
      src: slicedPineapple,
      caption: 'A sliced pineapple',
      category: 'Trained',
    },
    {
      src: bwStrawberry,
      caption: 'Black and white strawberry',
      category: 'Trained',
    },
    {
      src: heldBlackberries,
      caption: 'Held blackberries',
      category: 'Trained',
    },
    {
      src: untrainedApple,
      caption: 'An untrained apple',
      category: 'Untrained',
    },
    {
      src: untrainedWatermelon,
      caption: 'An untrained watermelon',
      category: 'Untrained',
    },
    {
      src: blackberriesRaspberries,
      caption: 'Blackberries along with untrained raspberries',
      category: 'Mixed',
    },
    {
      src: cartoonFruits,
      caption: 'Select cartoon fruits',
      category: 'Other',
    },
  ];

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '30px', textAlign: 'center' }}>
        <h2
          style={{
            marginBottom: '15px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            fontSize: '32px',
            fontWeight: 'bold',
          }}
        >
          <span
            style={{
              background: 'linear-gradient(135deg, #667eea, #764ba2)',
              color: 'white',
              width: '60px',
              height: '60px',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <AiOutlineInfoCircle size={32} />
          </span>
          Fruit Colorizer Explained
        </h2>
        <p style={{ color: '#666', fontSize: '18px', maxWidth: '800px', margin: '0 auto' }}>
          See how our AI model brings grayscale fruit images to life with accurate colors
        </p>
      </div>

      {/* Section Toggle */}
      <div
        style={{
          display: 'flex',
          gap: '10px',
          marginBottom: '30px',
          justifyContent: 'center',
          flexWrap: 'wrap',
        }}
      >
        <button
          onClick={() => setActiveSection('overview')}
          style={{
            padding: '12px 24px',
            fontSize: '16px',
            fontWeight: activeSection === 'overview' ? 'bold' : 'normal',
            color: activeSection === 'overview' ? 'white' : '#667eea',
            background: activeSection === 'overview' ? 'linear-gradient(135deg, #667eea, #764ba2)' : 'white',
            border: '2px solid #667eea',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.3s',
          }}
        >
          Overview & Examples
        </button>
        <button
          onClick={() => setActiveSection('technical')}
          style={{
            padding: '12px 24px',
            fontSize: '16px',
            fontWeight: activeSection === 'technical' ? 'bold' : 'normal',
            color: activeSection === 'technical' ? 'white' : '#667eea',
            background: activeSection === 'technical' ? 'linear-gradient(135deg, #667eea, #764ba2)' : 'white',
            border: '2px solid #667eea',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.3s',
          }}
        >
          Technical Details
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div
          style={{
            background: '#fee',
            border: '1px solid #fcc',
            borderRadius: '8px',
            padding: '12px 16px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <AiOutlineWarning size={20} color="#c33" />
          <span style={{ color: '#c33' }}>{error}</span>
        </div>
      )}

      {/* OVERVIEW SECTION */}
      {activeSection === 'overview' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
          {/* What Does It Do - Simple Introduction */}
          <div
            style={{
              background: 'linear-gradient(135deg, #667eea, #764ba2)',
              borderRadius: '16px',
              padding: '30px',
              color: 'white',
              textAlign: 'center',
            }}
          >
            <h3 style={{ marginTop: 0, marginBottom: '15px', fontSize: '28px', fontWeight: 'bold' }}>
              What Does This Model Do?
            </h3>
            <p style={{ fontSize: '18px', lineHeight: '1.8', margin: '0', maxWidth: '800px', marginLeft: 'auto', marginRight: 'auto' }}>
              This AI model takes a grayscale (black and white) image of a fruit and automatically adds realistic
              colors to it. It knows what color different fruits should be based on their shapes and textures!
            </p>
          </div>

          {/* Image Showcase Gallery */}
          <div
            style={{
              background: 'white',
              borderRadius: '16px',
              padding: '30px',
              boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
            }}
          >
            <h3
              style={{
                marginTop: 0,
                marginBottom: '10px',
                color: '#667eea',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                fontSize: '26px',
                fontWeight: 'bold',
              }}
            >
              <AiOutlinePlayCircle size={28} />
              See It In Action
            </h3>
            <p style={{ color: '#666', fontSize: '16px', marginBottom: '25px' }}>
              Here are real examples of what our model can do:
            </p>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
                gap: '25px',
              }}
            >
              {showcaseImages.map((image, idx) => (
                <div
                  key={idx}
                  style={{
                    background: '#f9f9f9',
                    borderRadius: '12px',
                    overflow: 'hidden',
                    border: '2px solid #e0e0e0',
                  }}
                >
                  <img
                    src={image.src}
                    alt={image.caption}
                    style={{
                      width: '100%',
                      height: 'auto',
                      display: 'block',
                    }}
                  />
                  <div style={{ padding: '15px', textAlign: 'center' }}>
                    <div style={{ fontSize: '16px', color: '#333', lineHeight: '1.6', marginBottom: '5px' }}>
                      {image.caption}
                    </div>
                    <div
                      style={{
                        fontSize: '13px',
                        color: 'white',
                        background: image.category === 'Trained' ? '#22c55e' : image.category === 'Untrained' ? '#667eea' : '#f59e0b',
                        padding: '4px 12px',
                        borderRadius: '12px',
                        display: 'inline-block',
                        fontWeight: 'bold',
                      }}
                    >
                      {image.category}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Trained Fruits */}
          <div
            style={{
              background: 'white',
              borderRadius: '16px',
              padding: '30px',
              boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
            }}
          >
            <h3
              style={{
                marginTop: 0,
                marginBottom: '10px',
                color: '#667eea',
                fontSize: '26px',
                fontWeight: 'bold',
              }}
            >
              What Fruits Was It Trained On?
            </h3>
            <p style={{ color: '#666', fontSize: '16px', marginBottom: '25px' }}>
              The model was specifically trained to recognize and colorize these 5 fruits:
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '20px' }}>
              {trainedFruits.map((fruit) => {
                const IconComponent = fruit.icon;
                return (
                  <div
                    key={fruit.name}
                    style={{
                      background: '#f9f9f9',
                      borderRadius: '12px',
                      padding: '25px 20px',
                      border: '2px solid #e0e0e0',
                      textAlign: 'center',
                    }}
                  >
                    <IconComponent size={50} color="#667eea" style={{ marginBottom: '15px' }} />
                    <div style={{ fontWeight: 'bold', fontSize: '20px', color: '#333', marginBottom: '8px' }}>
                      {fruit.name}
                    </div>
                    <div style={{ fontSize: '14px', color: '#666' }}>{fruit.description}</div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* How to Use */}
          <div
            style={{
              background: 'white',
              borderRadius: '16px',
              padding: '30px',
              boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
            }}
          >
            <h3
              style={{
                marginTop: 0,
                marginBottom: '10px',
                color: '#667eea',
                fontSize: '26px',
                fontWeight: 'bold',
              }}
            >
              <AiOutlineCheckCircle size={28} style={{ verticalAlign: 'middle', marginRight: '10px' }} />
              How to Use This Model
            </h3>
            <p style={{ color: '#666', fontSize: '16px', marginBottom: '20px' }}>
              Follow these simple steps to colorize your fruit images:
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '15px' }}>
                <div
                  style={{
                    minWidth: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    background: '#667eea',
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 'bold',
                    fontSize: '18px',
                    flexShrink: 0,
                  }}
                >
                  1
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 'bold', fontSize: '18px', color: '#333', marginBottom: '5px' }}>
                    Upload a grayscale image
                  </div>
                  <div style={{ color: '#666', fontSize: '16px', lineHeight: '1.6' }}>
                    Go to the "Fruit Colorization" tab and upload a black and white image of a fruit (works best with
                    bananas, blackberries, pineapples, oranges, or strawberries)
                  </div>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '15px' }}>
                <div
                  style={{
                    minWidth: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    background: '#667eea',
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 'bold',
                    fontSize: '18px',
                    flexShrink: 0,
                  }}
                >
                  2
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 'bold', fontSize: '18px', color: '#333', marginBottom: '5px' }}>
                    Select your model
                  </div>
                  <div style={{ color: '#666', fontSize: '16px', lineHeight: '1.6' }}>
                    Choose which trained model you want to use from the dropdown menu
                  </div>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '15px' }}>
                <div
                  style={{
                    minWidth: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    background: '#667eea',
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 'bold',
                    fontSize: '18px',
                    flexShrink: 0,
                  }}
                >
                  3
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 'bold', fontSize: '18px', color: '#333', marginBottom: '5px' }}>
                    Click "Colorize"
                  </div>
                  <div style={{ color: '#666', fontSize: '16px', lineHeight: '1.6' }}>
                    The AI will process your image and add realistic colors
                  </div>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '15px' }}>
                <div
                  style={{
                    minWidth: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    background: '#667eea',
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 'bold',
                    fontSize: '18px',
                    flexShrink: 0,
                  }}
                >
                  4
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 'bold', fontSize: '18px', color: '#333', marginBottom: '5px' }}>
                    View and download
                  </div>
                  <div style={{ color: '#666', fontSize: '16px', lineHeight: '1.6' }}>
                    See your colorized image side-by-side with the original, and download it if you like the results!
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Capabilities and Limitations */}
          <div
            style={{
              background: 'white',
              borderRadius: '16px',
              padding: '30px',
              boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
            }}
          >
            <h3 style={{ marginTop: 0, marginBottom: '25px', color: '#667eea', fontSize: '26px', fontWeight: 'bold' }}>
              What Can It Do? (And What Can't It Do?)
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '25px' }}>
              {/* Capabilities */}
              <div>
                <h4
                  style={{
                    color: '#22c55e',
                    marginTop: 0,
                    marginBottom: '15px',
                    fontSize: '20px',
                  }}
                >
                  <AiOutlineCheckCircle size={24} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
                  It's Great At:
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ color: '#22c55e', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>✓</div>
                    <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                      Colorizing the 5 trained fruits perfectly (bananas, blackberries, pineapples, oranges,
                      strawberries)
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ color: '#22c55e', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>✓</div>
                    <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                      Knowing the difference between fruit parts (leaves should be green, flesh should be the right
                      color)
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ color: '#22c55e', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>✓</div>
                    <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                      Working with different angles, lighting, and fruit arrangements
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ color: '#22c55e', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>✓</div>
                    <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                      Even coloring similar fruits it wasn't trained on (like apples, raspberries, and watermelons)
                    </div>
                  </div>
                </div>
              </div>
              {/* Limitations */}
              <div>
                <h4
                  style={{
                    color: '#ef4444',
                    marginTop: 0,
                    marginBottom: '15px',
                    fontSize: '20px',
                  }}
                >
                  <AiOutlineWarning size={24} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
                  Be Aware:
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ color: '#ef4444', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>✗</div>
                    <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                      May color non-fruit objects incorrectly (it thinks everything is a fruit!)
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ color: '#ef4444', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>✗</div>
                    <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                      Works best with simple backgrounds (complex scenes might look weird)
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ color: '#ef4444', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>✗</div>
                    <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                      Multiple overlapping fruits or partially hidden fruits can confuse it
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ color: '#ef4444', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>✗</div>
                    <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                      Unusual fruit varieties (like green strawberries) won't work as expected
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* How It Works - Simplified */}
          <div
            style={{
              background: 'white',
              borderRadius: '16px',
              padding: '30px',
              boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
            }}
          >
            <h3
              style={{
                marginTop: 0,
                marginBottom: '10px',
                color: '#667eea',
                fontSize: '26px',
                fontWeight: 'bold',
              }}
            >
              <AiOutlineBulb size={28} style={{ verticalAlign: 'middle', marginRight: '10px' }} />
              How Does It Work? (Simplified)
            </h3>
            <p style={{ color: '#666', fontSize: '16px', marginBottom: '25px' }}>
              Here's a simple explanation of how the AI figures out what colors to use:
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div
                style={{
                  background: '#f0f4ff',
                  borderRadius: '12px',
                  padding: '20px',
                  borderLeft: '4px solid #667eea',
                }}
              >
                <div style={{ fontWeight: 'bold', fontSize: '18px', color: '#667eea', marginBottom: '10px' }}>
                  Step 1: Learning Phase (Training)
                </div>
                <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.7' }}>
                  We showed the AI thousands of fruit images - both in color and grayscale. It learned patterns like:
                  "round and bumpy texture = orange color" or "elongated smooth shape = yellow color" or "pointy with
                  leafy top = red body with green leaves."
                </div>
              </div>
              <div
                style={{
                  background: '#f0f4ff',
                  borderRadius: '12px',
                  padding: '20px',
                  borderLeft: '4px solid #667eea',
                }}
              >
                <div style={{ fontWeight: 'bold', fontSize: '18px', color: '#667eea', marginBottom: '10px' }}>
                  Step 2: Recognition Phase
                </div>
                <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.7' }}>
                  When you upload a grayscale image, the AI looks at the shape and texture. It tries to match what it
                  sees with the patterns it learned. "This looks round and bumpy... it's probably an orange!"
                </div>
              </div>
              <div
                style={{
                  background: '#f0f4ff',
                  borderRadius: '12px',
                  padding: '20px',
                  borderLeft: '4px solid #667eea',
                }}
              >
                <div style={{ fontWeight: 'bold', fontSize: '18px', color: '#667eea', marginBottom: '10px' }}>
                  Step 3: Coloring Phase
                </div>
                <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.7' }}>
                  Once it recognizes the fruit, it adds the appropriate colors pixel by pixel. It doesn't just paint
                  the whole thing one color - it knows that orange peel is orange, the inside is lighter, leaves are
                  green, etc.
                </div>
              </div>
            </div>
          </div>

          {/* Why This Model vs Diffusion */}
          <div
            style={{
              background: 'white',
              borderRadius: '16px',
              padding: '30px',
              boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
            }}
          >
            <h3
              style={{
                marginTop: 0,
                marginBottom: '10px',
                color: '#667eea',
                fontSize: '26px',
                fontWeight: 'bold',
              }}
            >
              <AiOutlineThunderbolt size={28} style={{ verticalAlign: 'middle', marginRight: '10px' }} />
              Why This Specialized Model?
            </h3>
            <p style={{ color: '#333', fontSize: '16px', lineHeight: '1.7', marginBottom: '15px' }}>
              We previously built a diffusion model that could color images, but it had a problem: it could color based
              on shapes, but it didn't know <strong>what</strong> those shapes were or <strong>what colors</strong> to
              use.
            </p>
            <p style={{ color: '#333', fontSize: '16px', lineHeight: '1.7', marginBottom: '15px' }}>
              This U-Net model solves that problem by focusing specifically on fruits. Instead of trying to colorize
              anything and everything, it specializes in just 5 fruits. This makes it:
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                <div style={{ color: '#667eea', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>•</div>
                <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                  <strong>More accurate</strong> - It knows exactly what color strawberries should be
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                <div style={{ color: '#667eea', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>•</div>
                <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                  <strong>Better at details</strong> - Can color different fruit parts (peel, flesh, leaves)
                  correctly
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                <div style={{ color: '#667eea', fontSize: '20px', flexShrink: 0, lineHeight: '1.6' }}>•</div>
                <div style={{ color: '#333', fontSize: '16px', lineHeight: '1.6', flex: 1 }}>
                  <strong>More reliable</strong> - Consistent results on the fruits it was trained on
                </div>
              </div>
            </div>
            <div
              style={{
                background: '#f0f4ff',
                borderLeft: '4px solid #667eea',
                padding: '20px',
                borderRadius: '8px',
              }}
            >
              <div style={{ fontWeight: 'bold', color: '#667eea', marginBottom: '8px', fontSize: '16px' }}>
                Size Comparison:
              </div>
              <div style={{ color: '#333', fontSize: '15px', lineHeight: '1.6' }}>
                Diffusion model (colorizes anything): ~350 MB
              </div>
              <div style={{ color: '#333', fontSize: '15px', lineHeight: '1.6' }}>
                This U-Net model (specialized for fruits): ~740 MB
              </div>
              <div style={{ color: '#666', marginTop: '10px', fontSize: '14px', fontStyle: 'italic' }}>
                A model that could perfectly colorize ANY image would be exponentially larger and more complex!
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TECHNICAL SECTION */}
      {activeSection === 'technical' && (
        <div>
          {/* Model Selection */}
          <div style={{ marginBottom: '30px' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '8px',
                fontWeight: 'bold',
                color: '#333',
                fontSize: '16px',
              }}
            >
              Select Model
            </label>
            {loading ? (
              <div style={{ padding: '10px', color: '#666' }}>Loading models...</div>
            ) : models.length === 0 ? (
              <div style={{ padding: '10px', color: '#c33' }}>No models found. Please train a model first.</div>
            ) : (
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{
                  width: '100%',
                  padding: '12px',
                  fontSize: '16px',
                  borderRadius: '8px',
                  border: '2px solid #ddd',
                  background: 'white',
                }}
              >
                {models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Technical Deep Dive */}
          <div
            style={{
              background: 'white',
              borderRadius: '16px',
              padding: '30px',
              boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
              marginBottom: '25px',
            }}
          >
            <h3 style={{ marginTop: 0, color: '#667eea', fontSize: '24px', marginBottom: '15px' }}>
              U-Net Architecture Explained
            </h3>
            <div style={{ color: '#333', lineHeight: '1.8', fontSize: '16px' }}>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#667eea' }}>What is U-Net?</strong>
                <div style={{ marginTop: '8px' }}>
                  U-Net is a convolutional neural network architecture designed for image-to-image translation tasks.
                  It's called "U-Net" because the architecture looks like the letter "U" when drawn out.
                </div>
              </div>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#667eea' }}>Encoder (Downsampling Path):</strong>
                <div style={{ marginTop: '8px' }}>
                  The encoder compresses the input image through 4 layers, progressively reducing spatial dimensions
                  while increasing feature channels: 64 → 128 → 256 → 512 channels. This learns increasingly abstract
                  representations of the fruit's shape and texture.
                </div>
              </div>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#667eea' }}>Bottleneck:</strong>
                <div style={{ marginTop: '8px' }}>
                  The deepest layer (1024 channels) where the model learns the most complex patterns like "round bumpy
                  texture = orange" or "elongated shape = banana."
                </div>
              </div>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#667eea' }}>Decoder (Upsampling Path):</strong>
                <div style={{ marginTop: '8px' }}>
                  The decoder reconstructs the full-size color image through 4 layers: 512 → 256 → 128 → 64 channels,
                  finally outputting 3 RGB color channels.
                </div>
              </div>
              <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: '#667eea' }}>Skip Connections:</strong>
                <div style={{ marginTop: '8px' }}>
                  These are direct connections between encoder and decoder layers at the same level. They help preserve
                  fine details like fruit texture and leaf edges, ensuring the colorized output looks sharp and
                  accurate.
                </div>
              </div>
            </div>
          </div>

          {/* Model Information Cards */}
          {loadingInfo ? (
            <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>Loading model information...</div>
          ) : modelInfo ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {/* Basic Info Card */}
              <div
                style={{
                  background: 'white',
                  borderRadius: '12px',
                  padding: '20px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                }}
              >
                <h3 style={{ marginTop: 0, color: '#667eea' }}>Basic Information</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                  <div>
                    <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Model Name</div>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>{modelInfo.model_name}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Description</div>
                    <div style={{ fontSize: '16px', color: '#333' }}>
                      {modelInfo.description || 'U-Net based fruit colorizer'}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Image Size</div>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>
                      {modelInfo.image_size}x{modelInfo.image_size} pixels
                    </div>
                  </div>
                  {modelInfo.total_parameters && (
                    <div>
                      <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Total Parameters</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>
                        {modelInfo.total_parameters.toLocaleString()}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Architecture Card */}
              <div
                style={{
                  background: 'white',
                  borderRadius: '12px',
                  padding: '20px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                }}
              >
                <h3 style={{ marginTop: 0, color: '#667eea' }}>Model Architecture Specifications</h3>
                <div style={{ color: '#666', lineHeight: '1.8' }}>
                  <div>
                    <strong>Architecture:</strong> U-Net (Encoder-Decoder with Skip Connections)
                  </div>
                  <div>
                    <strong>Input:</strong> Grayscale images (1 channel, {modelInfo.image_size}x{modelInfo.image_size})
                  </div>
                  <div>
                    <strong>Output:</strong> RGB images (3 channels, {modelInfo.image_size}x{modelInfo.image_size})
                  </div>
                  <div>
                    <strong>Encoder Layers:</strong> 4 downsampling blocks (64→128→256→512 channels)
                  </div>
                  <div>
                    <strong>Bottleneck:</strong> 1024 channels
                  </div>
                  <div>
                    <strong>Decoder Layers:</strong> 4 upsampling blocks (512→256→128→64 channels)
                  </div>
                  <div>
                    <strong>Activation:</strong> ReLU (hidden layers), Sigmoid (output layer)
                  </div>
                </div>
              </div>

              {/* Training Config Card */}
              {modelInfo.training_config && (
                <div
                  style={{
                    background: 'white',
                    borderRadius: '12px',
                    padding: '20px',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  }}
                >
                  <h3 style={{ marginTop: 0, color: '#667eea' }}>Training Configuration</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    <div>
                      <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Batch Size</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>
                        {modelInfo.training_config.batch_size}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Learning Rate</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>
                        {modelInfo.training_config.learning_rate}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Optimizer</div>
                      <div style={{ fontSize: '16px', color: '#333' }}>
                        {modelInfo.training_config.optimizer.toUpperCase()}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Loss Function</div>
                      <div style={{ fontSize: '16px', color: '#333' }}>
                        {modelInfo.training_config.loss_function.toUpperCase()}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Total Epochs</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>
                        {modelInfo.training_config.epochs}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Created At</div>
                      <div style={{ fontSize: '16px', color: '#333' }}>
                        {new Date(modelInfo.training_config.created_at).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Training Stats Card */}
              {modelInfo.training_stats && (
                <div
                  style={{
                    background: 'white',
                    borderRadius: '12px',
                    padding: '20px',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  }}
                >
                  <h3 style={{ marginTop: 0, color: '#667eea' }}>Training Statistics</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    <div>
                      <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>Total Epochs Trained</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>
                        {modelInfo.training_stats.total_epochs}
                      </div>
                    </div>
                    {modelInfo.training_stats.best_val_loss && (
                      <div>
                        <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>
                          Best Validation Loss
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>
                          {modelInfo.training_stats.best_val_loss.toFixed(6)}
                        </div>
                      </div>
                    )}
                    {modelInfo.training_stats.final_train_loss && (
                      <div>
                        <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>
                          Final Training Loss
                        </div>
                        <div style={{ fontSize: '16px', color: '#333' }}>
                          {modelInfo.training_stats.final_train_loss.toFixed(6)}
                        </div>
                      </div>
                    )}
                    {modelInfo.training_stats.final_val_loss && (
                      <div>
                        <div style={{ fontSize: '14px', color: '#999', marginBottom: '5px' }}>
                          Final Validation Loss
                        </div>
                        <div style={{ fontSize: '16px', color: '#333' }}>
                          {modelInfo.training_stats.final_val_loss.toFixed(6)}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}

export default ModelInformation;
