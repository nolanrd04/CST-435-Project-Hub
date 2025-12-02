import React from 'react';
import { Link } from 'react-router-dom';
import { FaYoutube, FaGithub, FaBrain, FaImage, FaRobot, FaMusic, FaPalette, FaDna } from 'react-icons/fa';
import './HomePage.css';

interface Project {
  id: string;
  path: string;
  title: string;
  subtitle: string;
  description: string;
  icon: React.ReactNode;
  gradient: string;
  category: 'course' | 'personal';
  featured?: boolean;
}

const projects: Project[] = [
  {
    id: 'text-gen',
    path: '/text-generator',
    title: 'Text Generator',
    subtitle: 'LSTM Neural Network',
    description: 'Generate creative text using advanced recurrent neural networks with temperature-controlled sampling.',
    icon: <FaBrain />,
    gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    category: 'personal'
  },
  {
    id: 'genetic',
    path: '/genetic-algorithm',
    title: 'Genetic Algorithm',
    subtitle: 'Evolution Simulation',
    description: 'Watch evolutionary algorithms solve optimization problems through natural selection and mutation.',
    icon: <FaDna />,
    gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    category: 'personal'
  },
  {
    id: 'project3',
    path: '/image-classifier',
    title: 'Image Classifier',
    subtitle: 'Computer Vision',
    description: 'Classify images using convolutional neural networks trained on diverse datasets.',
    icon: <FaImage />,
    gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    category: 'course'
  },
  {
    id: 'project4',
    path: '/sentiment-analyzer',
    title: 'Sentiment Analyzer',
    subtitle: 'NLP Analysis',
    description: 'Analyze text sentiment and emotions using natural language processing techniques.',
    icon: <FaRobot />,
    gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    category: 'course'
  },
  {
    id: 'project5',
    path: '/Project5',
    title: 'Song Lyric Generator',
    subtitle: 'Music AI',
    description: 'Create original song lyrics using recurrent neural networks trained on musical datasets.',
    icon: <FaMusic />,
    gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
    category: 'course'
  },
  {
    id: 'project6',
    path: '/project6',
    title: 'Image Generator',
    subtitle: 'GAN Technology',
    description: 'Generate unique images using Generative Adversarial Networks and diffusion models.',
    icon: <FaPalette />,
    gradient: 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
    category: 'course',
    featured: true
  },
  {
    id: 'project7',
    path: '/project7',
    title: 'Image Colorization',
    subtitle: 'Deep Learning',
    description: 'Transform grayscale images into vibrant color photos using deep learning colorization.',
    icon: <FaPalette />,
    gradient: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
    category: 'personal',
    featured: true
  }
];

export default function HomePage() {
  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            <span className="gradient-text">CST435 Project Hub</span>
          </h1>
          <p className="hero-subtitle">
            Nolan and John's AI & Machine Learning Showcase
          </p>
          <p className="hero-description">
            Explore our collection of cutting-edge machine learning projects, from neural networks to generative AI
          </p>

          {/* External Links */}
          <div className="hero-links">
            <a
              href="https://youtu.be/wy0sXLL7oKw"
              target="_blank"
              rel="noopener noreferrer"
              className="hero-link youtube"
            >
              <FaYoutube size={24} />
              <span>Video Showcase</span>
            </a>
            <a
              href="https://github.com/nolanrd04/CST-435-Project-Hub"
              target="_blank"
              rel="noopener noreferrer"
              className="hero-link github"
            >
              <FaGithub size={24} />
              <span>GitHub Repository</span>
            </a>
          </div>
        </div>

        {/* Animated Background Elements */}
        <div className="hero-bg">
          <div className="floating-shape shape-1"></div>
          <div className="floating-shape shape-2"></div>
          <div className="floating-shape shape-3"></div>
          <div className="floating-shape shape-4"></div>
        </div>
      </section>

      {/* Course Projects Section */}
      <section className="projects-section">
        <div className="section-header">
          <h2 className="section-title">Course Projects</h2>
          <p className="section-description">Academic projects showcasing various ML techniques</p>
        </div>
        <div className="projects-grid">
          {projects.filter(p => p.category === 'course').map((project, index) => (
            <Link
              key={project.id}
              to={project.path}
              className={`project-card ${project.featured ? 'featured-project' : ''}`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {project.featured && <div className="featured-badge">Featured Project!</div>}
              <div className="project-card-inner">
                <div className="project-icon" style={{ background: project.gradient }}>
                  {project.icon}
                </div>
                <div className="project-content">
                  <h3 className="project-title">{project.title}</h3>
                  <p className="project-subtitle">{project.subtitle}</p>
                  <p className="project-description">{project.description}</p>
                </div>
                <div className="project-arrow">→</div>
              </div>
              <div className="project-card-bg" style={{ background: project.gradient }}></div>
            </Link>
          ))}
        </div>
      </section>
      {/* Personal Projects Section */}
      <section className="projects-section">
        <div className="section-header">
          <h2 className="section-title">Personal Projects (Nolan)</h2>
          <p className="section-description">Experimental and exploratory machine learning projects</p>
        </div>
        <div className="projects-grid">
          {projects.filter(p => p.category === 'personal').map((project, index) => (
            <Link
              key={project.id}
              to={project.path}
              className={`project-card ${project.featured ? 'featured-project' : ''}`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {project.featured && <div className="featured-badge">Featured Project!</div>}
              <div className="project-card-inner">
                <div className="project-icon" style={{ background: project.gradient }}>
                  {project.icon}
                </div>
                <div className="project-content">
                  <h3 className="project-title">{project.title}</h3>
                  <p className="project-subtitle">{project.subtitle}</p>
                  <p className="project-description">{project.description}</p>
                </div>
                <div className="project-arrow">→</div>
              </div>
              <div className="project-card-bg" style={{ background: project.gradient }}></div>
            </Link>
          ))}
        </div>
      </section>

      

      {/* Footer */}
      <footer className="home-footer">
        <p>Built with React, TypeScript, FastAPI, and PyTorch</p>
        <p className="footer-note">Projects increase in complexity and quality from Project 3 onwards</p>
      </footer>
    </div>
  );
}
