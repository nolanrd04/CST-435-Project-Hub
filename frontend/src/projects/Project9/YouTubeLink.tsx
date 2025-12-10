import React from 'react';
import { AiOutlineYoutube } from 'react-icons/ai';

function YouTubeLink() {
  // Update this URL when you have a YouTube video for Project 9
  const youtubeUrl = 'https://youtube.com';

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
          <AiOutlineYoutube size={24} />
        </span>
        YouTube Video
      </h2>

      <div
        style={{
          textAlign: 'center',
          padding: '40px',
          background: '#f9f9ff',
          borderRadius: '12px',
          border: '1px solid #e0e0ff',
        }}
      >
        <AiOutlineYoutube size={64} color="#667eea" style={{ marginBottom: '20px' }} />
        <h3 style={{ color: '#667eea', marginBottom: '15px' }}>
          Project 9: Fruit Image Colorization
        </h3>
        <p style={{ color: '#666', marginBottom: '25px', lineHeight: '1.6' }}>
          Watch our demonstration video showcasing the U-Net based image colorization model
          in action. See how our model transforms grayscale fruit images into vibrant,
          colorized versions.
        </p>
        <a
          href={youtubeUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            display: 'inline-block',
            padding: '14px 32px',
            fontSize: '16px',
            fontWeight: 'bold',
            color: 'white',
            background: 'linear-gradient(135deg, #667eea, #764ba2)',
            borderRadius: '8px',
            textDecoration: 'none',
            transition: 'all 0.3s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = '0 6px 20px rgba(102, 126, 234, 0.4)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          Watch on YouTube
        </a>
      </div>

      {/* Additional Info */}
      <div
        style={{
          marginTop: '30px',
          padding: '20px',
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}
      >
        <h4 style={{ marginTop: 0, color: '#667eea' }}>What you'll see in the video:</h4>
        <div style={{ color: '#666', lineHeight: '1.8' }}>
          <div>• Overview of the U-Net architecture used for colorization</div>
          <div>• Live demonstration of fruit image colorization</div>
          <div>• Technical challenges and solutions during development</div>
        </div>
      </div>
    </div>
  );
}

export default YouTubeLink;
