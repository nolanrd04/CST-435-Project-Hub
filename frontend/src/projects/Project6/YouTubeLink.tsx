import React from 'react';
import { AiOutlineYoutube } from 'react-icons/ai';

function YouTubeLink() {
  // Extract video ID from YouTube URL
  const videoUrl = 'https://youtu.be/nvfa5Tjtbpg';
  const videoId = videoUrl.split('/').pop()?.split('?')[0] || '';
  const embedUrl = `https://www.youtube.com/embed/${videoId}`;

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
          padding: '40px',
          maxWidth: '900px',
          margin: '0 auto',
        }}
      >
        {/* YouTube Embed */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            paddingBottom: '56.25%', // 16:9 aspect ratio
            height: 0,
            overflow: 'hidden',
            borderRadius: '8px',
            marginBottom: '30px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          }}
        >
          <iframe
            src={embedUrl}
            title="Project 6: GAN Image Generator Demonstration"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              border: 'none',
              borderRadius: '8px',
            }}
            allowFullScreen
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          />
        </div>

        <h3 style={{ fontSize: '20px', marginBottom: '15px', color: '#333', textAlign: 'center' }}>
          Project Demonstration
        </h3>
        <p style={{ fontSize: '14px', color: '#666', marginBottom: '20px', lineHeight: '1.6', textAlign: 'center' }}>
          Watch the comprehensive video walkthrough of this GAN image generation project covering:
        </p>
        <ul
          style={{
            listStyle: 'none',
            padding: 0,
            fontSize: '14px',
            color: '#666',
            lineHeight: '2',
            maxWidth: '600px',
            margin: '0 auto',
            textAlign: 'left',
          }}
        >
          <li>✓ Model training process and architecture</li>
          <li>✓ Image generation with different parameters</li>
          <li>✓ Quality comparison across different fruits</li>
          <li>✓ Cost analysis and performance metrics</li>
          <li>✓ Technical challenges and solutions</li>
        </ul>

        <div style={{ textAlign: 'center', marginTop: '25px' }}>
          <a
            href={videoUrl}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-block',
              padding: '12px 24px',
              backgroundColor: '#FF0000',
              color: 'white',
              textDecoration: 'none',
              borderRadius: '8px',
              fontWeight: 'bold',
              transition: 'background-color 0.3s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#CC0000')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#FF0000')}
          >
            Watch on YouTube
          </a>
        </div>
      </div>
    </div>
  );
}

export default YouTubeLink;
