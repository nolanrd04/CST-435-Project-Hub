import React from 'react';
import { AiOutlineYoutube } from 'react-icons/ai';

function YouTubeLink() {
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
            background: 'linear-gradient(135deg, #FF0000, #CC0000)',
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
          <AiOutlineYoutube size={24} />
        </span>
        YouTube Showcase
      </h2>

      <div
        style={{
          backgroundColor: '#f8f9ff',
          border: '2px solid #667eea',
          borderRadius: '12px',
          padding: '60px 40px',
          textAlign: 'center',
        }}
      >
        <div style={{ marginBottom: '20px' }}>
          <AiOutlineYoutube size={80} style={{ color: '#FF0000' }} />
        </div>
        <h3 style={{ fontSize: '24px', marginBottom: '15px', color: '#333' }}>
          Project Demonstration Coming Soon
        </h3>
        <p style={{ fontSize: '16px', color: '#666', marginBottom: '30px', lineHeight: '1.6' }}>
          A comprehensive video walkthrough of this GAN image generation project will be available
          here once completed. The video will demonstrate:
        </p>
        <ul
          style={{
            listStyle: 'none',
            padding: 0,
            fontSize: '14px',
            color: '#666',
            lineHeight: '2',
            maxWidth: '600px',
            margin: '0 auto 30px',
            textAlign: 'left',
          }}
        >
          <li>✓ Model training process and architecture</li>
          <li>✓ Image generation with different parameters</li>
          <li>✓ Quality comparison across different fruits</li>
          <li>✓ Cost analysis and performance metrics</li>
          <li>✓ Technical challenges and solutions</li>
        </ul>

        <div
          style={{
            padding: '20px',
            backgroundColor: '#f0f4ff',
            borderRadius: '8px',
            fontSize: '14px',
            color: '#667eea',
            fontStyle: 'italic',
            maxWidth: '500px',
            margin: '0 auto',
          }}
        >
          Check back soon for the complete project demonstration video!
        </div>
      </div>
    </div>
  );
}

export default YouTubeLink;
