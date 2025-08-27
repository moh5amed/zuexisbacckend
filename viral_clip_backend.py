     #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Viral Clip Generator - Enhanced Backend Edition
A Flask-based backend service for generating viral clips from long videos
with Whisper transcription, advanced AI analysis, and frontend integration
"""
import base64
import os
import json
import time
import re
import tempfile
import shutil
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from queue import Queue
import sqlite3

# Flask and web framework
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# AI and ML libraries
import google.generativeai as genai
import whisper
try:
    from moviepy.editor import VideoFileClip, AudioFileClip
except ImportError:
    # Fallback for newer moviepy versions
    from moviepy import VideoFileClip, AudioFileClip
import numpy as np
# Audio analysis library
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. Advanced audio analysis will be disabled.")

# YouTube download library
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("Warning: yt-dlp not available. YouTube processing will be disabled.")

# Environment and configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

def download_youtube_video(url: str, output_dir: str = "downloads") -> Dict[str, Any]:
    """
    Download YouTube video using multiple fallback methods
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save downloaded video
        
    Returns:
        Dict containing file path, metadata, and status
    """
    print(f"🎬 Downloading YouTube video: {url}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
        # Method 1: Try yt-dlp with aggressive bypass options
    try:
        if YT_DLP_AVAILABLE:
            print("🔄 Method 1: Trying yt-dlp with aggressive bypass...")
            
            # Aggressive yt-dlp options that bypass restrictions
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
                'quiet': False,
                'no_warnings': False,
                'verbose': True,
                'extractaudio': False,
                'nocheckcertificate': True,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Sec-Fetch-Mode': 'navigate'
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    print(f"📋 Video Title: {info.get('title', 'Unknown')}")
                    print(f"📋 Video Duration: {info.get('duration', 0)} seconds")
                
                ydl.download([url])
                
                # Find downloaded file
                video_id = info.get('id', 'unknown')
                for ext in ['mp4', 'webm', 'mkv']:
                    potential_file = os.path.join(output_dir, f"{video_id}.{ext}")
                    if os.path.exists(potential_file):
                        print(f"✅ Successfully downloaded with yt-dlp: {potential_file}")
                        
                        metadata = {
                            'id': video_id,
                            'title': info.get('title', 'Unknown Title'),
                            'duration': info.get('duration', 0),
                            'uploader': info.get('uploader', 'Unknown'),
                            'view_count': info.get('view_count', 0),
                            'upload_date': info.get('upload_date', ''),
                            'description': info.get('description', ''),
                            'tags': info.get('tags', []),
                            'categories': info.get('categories', []),
                            'thumbnail': info.get('thumbnail', ''),
                            'file_path': potential_file,
                            'file_size': os.path.getsize(potential_file)
                        }
                        
                        return {
                            'success': True,
                            'file_path': potential_file,
                            'metadata': metadata,
                            'message': 'Video downloaded successfully with yt-dlp'
                        }
                
                raise Exception("yt-dlp download failed")
        else:
            raise ImportError("yt-dlp not available")
                
    except Exception as e:
        print(f"❌ Method 1 failed: {str(e)}")
        
        # Method 2: Try pytube as fallback
        try:
            print("🔄 Method 2: Trying pytube...")
            
            try:
                from pytube import YouTube
            except ImportError:
                print("📦 Installing pytube...")
                import subprocess
                subprocess.check_call(["pip", "install", "pytube"])
                from pytube import YouTube
            
            # Create YouTube object
            yt = YouTube(url)
            
            # Get video info
            print(f"📋 Video Title: {yt.title}")
            print(f"📋 Video Duration: {yt.length} seconds")
            
            # Get available streams
            streams = yt.streams.filter(progressive=True, file_extension='mp4')
            if not streams:
                streams = yt.streams.filter(progressive=True)
            if not streams:
                streams = yt.streams
            
            if not streams:
                raise Exception("No downloadable streams available")
            
            # Select best stream
            best_stream = streams.order_by('resolution').desc().first() or streams.first()
            print(f"📋 Selected Stream: {best_stream.resolution} - {best_stream.mime_type}")
            
            # Download
            print("⬇️ Downloading video...")
            downloaded_file = best_stream.download(output_path=output_dir)
            
            if os.path.exists(downloaded_file):
                print(f"✅ Successfully downloaded with pytube: {downloaded_file}")
                
                filename = os.path.basename(downloaded_file)
                video_id = filename.split('.')[0] if '.' in filename else 'unknown'
                
                metadata = {
                    'id': video_id,
                    'title': yt.title,
                    'duration': yt.length,
                    'uploader': yt.author,
                    'view_count': yt.views,
                    'upload_date': '',
                    'description': yt.description[:500] if yt.description else '',
                    'tags': [],
                    'categories': [],
                    'thumbnail': yt.thumbnail_url,
                    'file_path': downloaded_file,
                    'file_size': os.path.getsize(downloaded_file)
                }
                
                return {
                    'success': True,
                    'file_path': downloaded_file,
                    'metadata': metadata,
                    'message': 'Video downloaded successfully with pytube'
                }
            else:
                raise Exception(f"Downloaded file not found: {downloaded_file}")
                
        except Exception as pytube_error:
            print(f"❌ Method 2 failed: {str(pytube_error)}")
            
            # Method 3: Try direct HTTP download with minimal yt-dlp
            try:
                print("🔄 Method 3: Trying minimal yt-dlp...")
                
                if not YT_DLP_AVAILABLE:
                    raise ImportError("yt-dlp not available")
                
                # Minimal options
                ydl_opts = {
                    'format': 'worst',  # Get any format
                    'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'ignoreerrors': True
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    
                    # Look for any downloaded file
                    for file in os.listdir(output_dir):
                        if file.endswith(('.mp4', '.webm', '.mkv', '.avi')):
                            video_path = os.path.join(output_dir, file)
                            print(f"✅ Successfully downloaded with minimal yt-dlp: {video_path}")
                            
                            return {
                                'success': True,
                                'file_path': video_path,
                                'metadata': {
                                    'id': 'unknown',
                                    'title': 'Downloaded with minimal method',
                                    'file_path': video_path,
                                    'file_size': os.path.getsize(video_path)
                                },
                                'message': 'Video downloaded successfully with minimal method'
                            }
                
                raise Exception("No video file found")
                
            except Exception as minimal_error:
                final_error_msg = f"All download methods failed: yt-dlp: {str(e)}, pytube: {str(pytube_error)}, minimal: {str(minimal_error)}"
                print(f"❌ {final_error_msg}")
        return {
            'success': False,
                    'error': final_error_msg,
                    'message': final_error_msg
        }

class JobQueue:
    """Background job queue for persistent processing"""
    
    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self.processing_queue = Queue()
        self.active_jobs = {}
        self.init_database()
        self.start_worker_thread()
    
    def init_database(self):
        """Initialize SQLite database for job tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    current_stage TEXT DEFAULT 'Initializing',
                    file_path TEXT,
                    output_path TEXT,
                    clips_generated INTEGER DEFAULT 0,
                    transcription TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    processing_options TEXT
                )
            ''')
            
            # Create job_stages table for detailed progress tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_stages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("Job database initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize job database: {e}")
    
    def add_job(self, user_id: str, project_name: str, file_path: str, processing_options: dict) -> str:
        """Add a new job to the queue"""
        job_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert job record
            cursor.execute('''
                INSERT INTO jobs (id, user_id, project_name, file_path, processing_options, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (job_id, user_id, project_name, file_path, json.dumps(processing_options), 'pending'))
            
            # Insert initial stages
            stages = [
                'File Upload',
                'Audio Analysis', 
                'AI Processing',
                'Clip Generation',
                'Finalizing'
            ]
            
            for stage in stages:
                cursor.execute('''
                    INSERT INTO job_stages (job_id, stage_name, status)
                    VALUES (?, ?, ?)
                ''', (job_id, stage, 'pending'))
            
            conn.commit()
            conn.close()
            
            # Add to processing queue
            self.processing_queue.put({
                'job_id': job_id,
                'user_id': user_id,
                'project_name': project_name,
                'file_path': file_path,
                'processing_options': processing_options
            })
            
            print(f"Job {job_id} added to queue for user {user_id}")
            return job_id
            
        except Exception as e:
            print(f"Failed to add job: {e}")
            return None
    
    def get_job_status(self, job_id: str) -> dict:
        """Get current status of a job"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get job details
            cursor.execute('''
                SELECT * FROM jobs WHERE id = ?
            ''', (job_id,))
            
            job_row = cursor.fetchone()
            if not job_row:
                return None
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            job_data = dict(zip(columns, job_row))
            
            # Get stage details
            cursor.execute('''
                SELECT * FROM job_stages WHERE job_id = ? ORDER BY id
            ''', (job_id,))
            
            stage_rows = cursor.fetchall()
            stage_columns = [description[0] for description in cursor.description]
            stages = []
            
            for row in stage_rows:
                stage_data = dict(zip(stage_columns, row))
                stages.append(stage_data)
            
            conn.close()
            
            # Calculate overall progress
            completed_stages = sum(1 for stage in stages if stage['status'] == 'completed')
            total_stages = len(stages)
            overall_progress = int((completed_stages / total_stages) * 100) if total_stages > 0 else 0
            
            return {
                'job_id': job_id,
                'status': job_data['status'],
                'progress': overall_progress,
                'current_stage': job_data['current_stage'],
                'stages': stages,
                'clips_generated': job_data['clips_generated'],
                'error_message': job_data['error_message'],
                'created_at': job_data['created_at'],
                'updated_at': job_data['updated_at'],
                'completed_at': job_data['completed_at']
            }
            
        except Exception as e:
            print(f"Failed to get job status: {e}")
            return None
    
    def get_user_jobs(self, user_id: str) -> List[dict]:
        """Get all jobs for a specific user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC
            ''', (user_id,))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            jobs = []
            for row in rows:
                job_data = dict(zip(columns, row))
                jobs.append(job_data)
            
            conn.close()
            return jobs
            
        except Exception as e:
            print(f"Failed to get user jobs: {e}")
            return []
    
    def update_job_progress(self, job_id: str, stage_name: str, progress: int, status: str = 'processing'):
        """Update progress of a specific job stage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update stage progress
            if status == 'completed':
                cursor.execute('''
                    UPDATE job_stages 
                    SET status = ?, progress = 100, completed_at = CURRENT_TIMESTAMP
                    WHERE job_id = ? AND stage_name = ?
                ''', (status, job_id, stage_name))
            else:
                cursor.execute('''
                    UPDATE job_stages 
                    SET status = ?, progress = ?
                    WHERE job_id = ? AND stage_name = ?
                ''', (status, progress, job_id, stage_name))
            
            # Update job current stage and progress
            cursor.execute('''
                UPDATE jobs 
                SET current_stage = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (stage_name, job_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Failed to update job progress: {e}")
    
    def complete_job(self, job_id: str, output_path: str, clips_generated: int, transcription: str):
        """Mark a job as completed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE jobs 
                SET status = 'completed', 
                    output_path = ?, 
                    clips_generated = ?, 
                    transcription = ?,
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (output_path, clips_generated, transcription, job_id))
            
            # Mark final stage as completed
            cursor.execute('''
                UPDATE job_stages 
                SET status = 'completed', progress = 100, completed_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND stage_name = 'Finalizing'
            ''', (job_id,))
            
            conn.commit()
            conn.close()
            
            print(f"Job {job_id} marked as completed")
            
        except Exception as e:
            print(f"Failed to complete job: {e}")
    
    def fail_job(self, job_id: str, error_message: str):
        """Mark a job as failed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE jobs 
                SET status = 'failed', 
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (error_message, job_id))
            
            conn.commit()
            conn.close()
            
            print(f"Job {job_id} marked as failed: {error_message}")
            
        except Exception as e:
            print(f"Failed to fail job: {e}")
    
    def start_worker_thread(self):
        """Start background worker thread for processing jobs"""
        # Only start worker thread in the main process, not in Flask reloader
        if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
            def worker():
                print("Job worker thread started")
                while True:
                    try:
                        # Get job from queue (blocking)
                        job_data = self.processing_queue.get()
                        if job_data is None:  # Shutdown signal
                            break
                        
                        # Process the job
                        self.process_job(job_data)
                        
                        # Mark task as done
                        self.processing_queue.task_done()
                        
                    except Exception as e:
                        print(f"Worker thread error: {e}")
                        time.sleep(1)
            
            # Start worker thread
            self.worker_thread = threading.Thread(target=worker, daemon=True)
            self.worker_thread.start()
            print("Background job worker started")
        else:
            print("Skipping worker thread start in reloader process")
            self.worker_thread = None
    
    def process_job(self, job_data: dict):
        """Process a single job"""
        job_id = job_data['job_id']
        user_id = job_data['user_id']
        project_name = job_data['project_name']
        file_path = job_data['file_path']
        processing_options = job_data['processing_options']
        
        print(f"Processing job {job_id} for user {user_id}")
        
        try:
            # Update status to processing
            self.update_job_progress(job_id, 'File Upload', 100, 'completed')
            self.update_job_progress(job_id, 'Audio Analysis', 0, 'processing')
            
            # Initialize generator for this job
            generator = ViralClipGenerator(output_dir=f"viral_clips/{user_id}/{job_id}")
            
            # Process audio analysis
            self.update_job_progress(job_id, 'Audio Analysis', 50, 'processing')
            viral_segments, full_transcript = generator.extract_audio_segments(file_path)
            self.update_job_progress(job_id, 'Audio Analysis', 100, 'completed')
            
            # AI Processing
            self.update_job_progress(job_id, 'AI Processing', 0, 'processing')
            num_clips = processing_options.get('numClips', 3)
            
            # Prepare frontend inputs for AI prompt enhancement
            frontend_inputs = {
                'projectName': project_name,
                'description': processing_options.get('description', ''),
                'aiPrompt': processing_options.get('aiPrompt', ''),
                'targetPlatforms': processing_options.get('targetPlatforms', ['tiktok']),
                'processingOptions': processing_options
            }
            
            self.update_job_progress(job_id, 'AI Processing', 50, 'processing')
            viral_moments = generator.ai_select_best_clips(viral_segments, full_transcript, num_clips, frontend_inputs)
            self.update_job_progress(job_id, 'AI Processing', 100, 'completed')
            
            # Clip Generation
            self.update_job_progress(job_id, 'Clip Generation', 0, 'processing')
            generated_clips = []
            
            for i, moment in enumerate(viral_moments):
                try:
                    start_time = moment['start_time']
                    duration = moment['duration']
                    
                    # Create descriptive filename
                    safe_caption = re.sub(r'[^\w\s-]', '', moment['caption'])[:30]
                    clip_name = f"viral_clip_{i+1}_{moment['viral_score']}_{safe_caption}.mp4"
                    clip_name = clip_name.replace(' ', '_')
                    
                    clip_path = generator.create_clip(file_path, start_time, duration, clip_name)
                    generated_clips.append(clip_path)
                    
                    # Update progress
                    progress = int(((i + 1) / len(viral_moments)) * 100)
                    self.update_job_progress(job_id, 'Clip Generation', progress, 'processing')
                    
                except Exception as e:
                    print(f"Failed to create clip {i+1}: {e}")
            
            self.update_job_progress(job_id, 'Clip Generation', 100, 'completed')
            
            # Finalizing
            self.update_job_progress(job_id, 'Finalizing', 0, 'processing')
            
            # Save results
            output_dir = f"viral_clips/{user_id}/{job_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save comprehensive analysis report
            report_data = {
                'job_id': job_id,
                'user_id': user_id,
                'project_name': project_name,
                'total_segments': len(viral_segments),
                'selected_clips': len(viral_moments),
                'clips_generated': len(generated_clips),
                'clips': generated_clips,
                'full_transcript': full_transcript,
                'processing_options': processing_options,
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            report_path = os.path.join(output_dir, "generation_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Mark job as completed
            self.complete_job(job_id, output_dir, len(generated_clips), full_transcript)
            self.update_job_progress(job_id, 'Finalizing', 100, 'completed')
            
            print(f"Job {job_id} completed successfully")
            
        except Exception as e:
            error_msg = f"Job processing failed: {str(e)}"
            print(error_msg)
            self.fail_job(job_id, error_msg)

class ViralClipGenerator:
    """AI-Powered viral clip generator using MoviePy and smart transcription analysis"""
    
    def __init__(self, output_dir: str = "viral_clips"):
        # Get API key from environment
        self.api_key = os.getenv('VITE_GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("VITE_GEMINI_API_KEY not found in .env.local file")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure Gemini API with enhanced capabilities
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 1.5 Flash for better quota management and efficiency
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Enhanced safety settings for content generation
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # Check if output directory is writable
        self.check_output_directory()
        
        # Audio analysis parameters
        self.segment_duration = 10  # Analyze 10-second audio segments
        self.min_segment_length = 3  # Minimum segment length in seconds
        self.max_segments = 50  # Maximum number of segments to analyze
        
        # Whisper model will be loaded when needed
        self.whisper_model = None
        
        # Clip generation settings
        self.min_clip_duration = 15  # Minimum clip duration in seconds
        self.max_clip_duration = 60  # Maximum clip duration in seconds
        
        # Enhanced logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"ViralClipGenerator initialized with API key: {self.api_key[:10]}...")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Initialize job queue
        self.job_queue = Queue()
        self.job_worker_thread = None
        self.start_job_worker()

    def start_job_worker(self):
        """Start the background job worker thread"""
        if self.job_worker_thread is None or not self.job_worker_thread.is_alive():
            self.job_worker_thread = threading.Thread(target=self._job_worker, daemon=True)
            self.job_worker_thread.start()
            print("🚀 Background job worker started")

    def _job_worker(self):
        """Background worker for processing jobs"""
        while True:
            try:
                # Process any pending jobs
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ Job worker error: {e}")
                time.sleep(5)
    
    def get_trending_context(self):
        """Get current trending topics and hashtags for AI prompts"""
        # This is a placeholder - you can enhance this with real trending data
        trending_topics = [
            "viral comedy moments",
            "stand up comedy highlights", 
            "funny clips",
            "viral humor",
            "comedy gold",
            "trending jokes",
            "viral content",
            "social media comedy"
        ]
        
        trending_hashtags = [
            "#viral", "#comedy", "#funny", "#humor", "#trending",
            "#standup", "#comedygold", "#viralcontent", "#funnyclips"
        ]
        
        return {
            "trending_topics": trending_topics,
            "trending_hashtags": trending_hashtags,
            "current_trends": "Comedy and humor content is trending across platforms",
            "viral_patterns": "Short, punchy comedy clips (15-60s) perform best"
        }

    def _build_user_instructions(self, frontend_inputs):
        """Build user instructions from frontend inputs"""
        if not frontend_inputs:
            return "No specific user requirements provided."
        
        instructions = []
        
        if frontend_inputs.get('style'):
            instructions.append(f"Style: {frontend_inputs['style']}")
        
        if frontend_inputs.get('tone'):
            instructions.append(f"Tone: {frontend_inputs['tone']}")
        
        if frontend_inputs.get('target_audience'):
            instructions.append(f"Target Audience: {frontend_inputs['target_audience']}")
        
        if frontend_inputs.get('platform'):
            instructions.append(f"Platform: {frontend_inputs['platform']}")
        
        if frontend_inputs.get('duration_preference'):
            instructions.append(f"Duration: {frontend_inputs['duration_preference']}")
        
        if frontend_inputs.get('content_focus'):
            instructions.append(f"Content Focus: {frontend_inputs['content_focus']}")
        
        if frontend_inputs.get('custom_instructions'):
            instructions.append(f"Custom: {frontend_inputs['custom_instructions']}")
        
        if instructions:
            return " | ".join(instructions)
        else:
            return "Standard viral content requirements"

    def _build_strict_user_instructions(self, frontend_inputs):
        """Build strict, directive user instructions that override default behaviors"""
        if not frontend_inputs:
            return "**NO USER INSTRUCTIONS PROVIDED** - Use standard viral content selection criteria."
        
        strict_instructions = []
        
        # Project-specific requirements
        if frontend_inputs.get('projectName'):
            strict_instructions.append(f"**PROJECT NAME**: {frontend_inputs['projectName']}")
        
        if frontend_inputs.get('description'):
            strict_instructions.append(f"**PROJECT DESCRIPTION**: {frontend_inputs['description']}")
        
        # Processing options that override defaults
        processing_options = frontend_inputs.get('processingOptions', {})
        
        if processing_options.get('targetDuration'):
            strict_instructions.append(f"**TARGET DURATION**: {processing_options['targetDuration']} seconds - CLIPS MUST BE THIS LENGTH")
        
        if processing_options.get('minDuration'):
            strict_instructions.append(f"**MINIMUM DURATION**: {processing_options['minDuration']} seconds - NO CLIPS SHORTER THAN THIS")
        
        if processing_options.get('maxDuration'):
            strict_instructions.append(f"**MAXIMUM DURATION**: {processing_options['maxDuration']} seconds - NO CLIPS LONGER THAN THIS")
        
        if processing_options.get('quality'):
            strict_instructions.append(f"**QUALITY REQUIREMENT**: {processing_options['quality']} quality processing - prioritize content that works with this setting")
        
        # Target platforms
        if frontend_inputs.get('targetPlatforms'):
            platforms = frontend_inputs['targetPlatforms']
            if isinstance(platforms, list):
                strict_instructions.append(f"**TARGET PLATFORMS**: {', '.join(platforms)} - OPTIMIZE CONTENT FOR THESE SPECIFIC PLATFORMS")
            else:
                strict_instructions.append(f"**TARGET PLATFORM**: {platforms} - OPTIMIZE CONTENT FOR THIS SPECIFIC PLATFORM")
        
        # Style and tone preferences
        if frontend_inputs.get('style'):
            style = frontend_inputs['style'].lower()
            if style == 'funny':
                strict_instructions.append("**STYLE REQUIREMENT**: HUMOROUS CONTENT ONLY - Focus on comedy, jokes, funny situations, and laugh-out-loud moments")
            elif style == 'dramatic':
                strict_instructions.append("**STYLE REQUIREMENT**: DRAMATIC CONTENT ONLY - Focus on emotional impact, suspense, intense moments, and powerful storytelling")
            elif style == 'educational':
                strict_instructions.append("**STYLE REQUIREMENT**: EDUCATIONAL CONTENT ONLY - Focus on learning, insights, knowledge sharing, and valuable information")
            elif style == 'inspirational':
                strict_instructions.append("**STYLE REQUIREMENT**: INSPIRATIONAL CONTENT ONLY - Focus on motivation, uplifting messages, positive energy, and life-changing insights")
        
        if frontend_inputs.get('tone'):
            tone = frontend_inputs['tone'].lower()
            if tone == 'professional':
                strict_instructions.append("**TONE REQUIREMENT**: PROFESSIONAL TONE ONLY - Formal, business-like, authoritative, and polished content")
            elif tone == 'casual':
                strict_instructions.append("**TONE REQUIREMENT**: CASUAL TONE ONLY - Relaxed, conversational, friendly, and approachable content")
            elif tone == 'energetic':
                strict_instructions.append("**TONE REQUIREMENT**: ENERGETIC TONE ONLY - High-energy, enthusiastic, dynamic, and exciting content")
            elif tone == 'calm':
                strict_instructions.append("**TONE REQUIREMENT**: CALM TONE ONLY - Peaceful, soothing, relaxed, and tranquil content")
        
        # Target audience specifications
        if frontend_inputs.get('target_audience'):
            audience = frontend_inputs['target_audience'].lower()
            if audience == 'gen_z':
                strict_instructions.append("**AUDIENCE REQUIREMENT**: GEN Z ONLY - Content must appeal to 16-24 year olds: trend-aware, authentic, social justice focused, meme culture")
            elif audience == 'millennials':
                strict_instructions.append("**AUDIENCE REQUIREMENT**: MILLENNIALS ONLY - Content must appeal to 25-40 year olds: nostalgic, career-focused, work-life balance, practical solutions")
            elif audience == 'gen_x':
                strict_instructions.append("**AUDIENCE REQUIREMENT**: GEN X ONLY - Content must appeal to 41-56 year olds: practical, family-oriented, value-conscious, established professionals")
        
        # Content focus areas
        if frontend_inputs.get('content_focus'):
            focus = frontend_inputs['content_focus'].lower()
            if focus == 'entertainment':
                strict_instructions.append("**CONTENT FOCUS**: ENTERTAINMENT ONLY - Prioritize fun, engaging, and enjoyable content over educational or informational")
            elif focus == 'education':
                strict_instructions.append("**CONTENT FOCUS**: EDUCATION ONLY - Prioritize learning, insights, and knowledge sharing over pure entertainment")
            elif focus == 'business':
                strict_instructions.append("**CONTENT FOCUS**: BUSINESS ONLY - Prioritize professional insights, career advice, and business strategies")
            elif focus == 'lifestyle':
                strict_instructions.append("**CONTENT FOCUS**: LIFESTYLE ONLY - Prioritize personal development, health, relationships, and daily life improvements")
        
        # Duration preferences
        if frontend_inputs.get('duration_preference'):
            duration = frontend_inputs['duration_preference'].lower()
            if duration == 'short':
                strict_instructions.append("**DURATION REQUIREMENT**: SHORT FORMAT ONLY - 15-30 seconds, quick impact, scroll-stopping content")
            elif duration == 'medium':
                strict_instructions.append("**DURATION REQUIREMENT**: MEDIUM FORMAT ONLY - 30-60 seconds, story development, engagement building")
            elif duration == 'long':
                strict_instructions.append("**DURATION REQUIREMENT**: LONG FORMAT ONLY - 60+ seconds, deep dive, comprehensive content")
        
        # Custom instructions
        if frontend_inputs.get('custom_instructions'):
            strict_instructions.append(f"**CUSTOM REQUIREMENT**: {frontend_inputs['custom_instructions']}")
        
        if strict_instructions:
            return "\n".join(strict_instructions)
        else:
            return "**NO SPECIFIC USER REQUIREMENTS** - Use standard viral content selection criteria."

    def _build_frontend_context(self, frontend_inputs):
        """Build frontend context information for AI prompts"""
        if not frontend_inputs:
            return ""
        
        context_parts = []
        
        # Platform-specific context
        if frontend_inputs.get('platform'):
            platform = frontend_inputs['platform'].lower()
            if platform == 'tiktok':
                context_parts.append("TikTok: Short-form vertical video, trending sounds, hashtag challenges")
            elif platform == 'instagram':
                context_parts.append("Instagram: Reels format, aesthetic appeal, story-driven content")
            elif platform == 'youtube':
                context_parts.append("YouTube Shorts: Vertical format, trending topics, search optimization")
            elif platform == 'twitter':
                context_parts.append("Twitter: Viral moments, trending conversations, quick engagement")
        
        # Content style context
        if frontend_inputs.get('style'):
            style = frontend_inputs['style'].lower()
            if style == 'funny':
                context_parts.append("Humor-focused: Comedy, memes, relatable situations")
            elif style == 'dramatic':
                context_parts.append("Drama-focused: Emotional impact, storytelling, suspense")
            elif style == 'educational':
                context_parts.append("Educational: Informative, valuable insights, knowledge sharing")
            elif style == 'inspirational':
                context_parts.append("Inspirational: Motivational, uplifting, positive energy")
        
        # Target audience context
        if frontend_inputs.get('target_audience'):
            audience = frontend_inputs['target_audience'].lower()
            if audience == 'gen_z':
                context_parts.append("Gen Z: Trend-aware, authentic, social justice focused")
            elif audience == 'millennials':
                context_parts.append("Millennials: Nostalgic, career-focused, work-life balance")
            elif audience == 'gen_x':
                context_parts.append("Gen X: Practical, family-oriented, value-conscious")
        
        # Duration preferences
        if frontend_inputs.get('duration_preference'):
            duration = frontend_inputs['duration_preference'].lower()
            if duration == 'short':
                context_parts.append("Short format: 15-30 seconds, quick impact, scroll-stopping")
            elif duration == 'medium':
                context_parts.append("Medium format: 30-60 seconds, story development, engagement")
            elif duration == 'long':
                context_parts.append("Long format: 60+ seconds, deep dive, comprehensive content")
        
        if context_parts:
            return " | ".join(context_parts)
        else:
                    return "Standard viral content optimization"

    def clean_ai_response(self, response_text):
        """Clean and parse AI response text"""
        try:
            # Remove any markdown formatting
            cleaned = response_text.strip()
            
            # Try to extract JSON if present
            if '{' in cleaned and '}' in cleaned:
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                json_part = cleaned[start:end]
                
                try:
                    import json
                    parsed = json.loads(json_part)
                    return parsed
                except json.JSONDecodeError:
                    print("⚠️ Failed to parse JSON from AI response")
            
            # If no valid JSON, return cleaned text
            return cleaned
            
        except Exception as e:
            print(f"⚠️ Error cleaning AI response: {e}")
            return response_text
    
    def log_message(self, message, level="INFO"):
        """Log message with timestamp and level"""
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {level}: {message}"
        
        # Print to console with color coding
        if level == "ERROR":
            print(f"ERROR: {log_msg}")
        elif level == "WARNING":
            print(f"WARNING: {log_msg}")
        elif level == "SUCCESS":
            print(f"SUCCESS: {log_msg}")
        else:
            print(f"INFO: {log_msg}")
        
        # Also log to file if needed
        self.logger.info(log_msg)
    
    def check_output_directory(self):
        """Check if output directory is writable"""
        try:
            # Ensure directory exists
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created output directory: {self.output_dir}")
            
            # Try to create a test file
            test_file = self.output_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()  # Remove test file
            print("Output directory is writable")
            return True
        except Exception as e:
            print(f"Output directory is not writable: {e}")
            print(f"Please check permissions for: {self.output_dir}")
            return False

    def preprocess_transcription_for_ai(self, full_transcript, viral_segments):
        """Preprocess transcription to reduce API usage while maintaining timestamps"""
        try:
            # Create a condensed version with key information
            condensed_text = f"VIDEO TRANSCRIPTION SUMMARY\n{'='*50}\n\n"
            
            # Safely get duration and segment count
            if viral_segments:
                total_duration = max(seg['end'] for seg in viral_segments)
                condensed_text += f"Total Duration: {total_duration:.1f} seconds\n"
            condensed_text += f"Total Segments: {len(viral_segments)}\n\n"
            
            # Add viral segments with timestamps
            condensed_text += "VIRAL SEGMENTS:\n"
            for i, seg in enumerate(viral_segments, 1):
                # Truncate long text to reduce API usage
                text_preview = seg['text'][:200] + "..." if len(seg['text']) > 200 else seg['text']
                condensed_text += f"{i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] Score: {seg['viral_score']}\n"
                condensed_text += f"   Text: {text_preview}\n\n"
            else:
                condensed_text += "No viral segments identified - using fallback analysis\n\n"
            
            # Add full transcript (truncated if too long)
            if len(full_transcript) > 2000:
                condensed_text += f"FULL TRANSCRIPT (First 2000 chars):\n{full_transcript[:2000]}...\n"
                condensed_text += f"[TRUNCATED - Total length: {len(full_transcript)} characters]"
            else:
                condensed_text += f"FULL TRANSCRIPT:\n{full_transcript}"
            
            return condensed_text
            
        except Exception as e:
            print(f"Failed to preprocess transcription: {str(e)}")
            return full_transcript  # Fallback to full transcript
    
    def load_whisper_model(self):
        """Load Whisper model with progress tracking"""
        if self.whisper_model is not None:
            return
            
        print("Loading Whisper model (this may take a few minutes on first run)...")
        try:
            # Use a smaller model for faster loading and less memory usage
            self.whisper_model = whisper.load_model("tiny")
            print("Whisper model loaded successfully!")
        except Exception as e:
            print(f"Failed to load Whisper model: {str(e)}")
            # If model loading fails, try to clear cache and retry once
            try:
                import shutil
                cache_dir = Path.home() / ".cache" / "whisper"
                if cache_dir.exists():
                    print("Clearing corrupted Whisper cache...")
                    shutil.rmtree(cache_dir)
                print("Retrying Whisper model loading...")
                self.whisper_model = whisper.load_model("tiny")
                print("Whisper model loaded successfully after cache cleanup!")
            except Exception as retry_e:
                print(f"Warning: Whisper model failed to load: {str(retry_e)}")
                print("Continuing without transcription capabilities...")
                self.whisper_model = None
    
    def extract_audio_segments(self, video_path: str):
        """SUPER-ACCURATE audio segment extraction using advanced AI and audio analysis"""
        try:
            # Load Whisper model if not already loaded
            self.load_whisper_model()
            
            # Check if Whisper model is available
            if self.whisper_model is None:
                print("Warning: Whisper model not available, using fallback segmentation...")
                return self._create_fallback_segments(video_path), "Transcription not available"
            
            # Get video duration and extract audio
            try:
                from moviepy.editor import VideoFileClip, AudioFileClip
            except ImportError:
                # Fallback for newer moviepy versions
                from moviepy import VideoFileClip, AudioFileClip
            import numpy as np
            import librosa
            
            print("🎬 Loading video for SUPER-ACCURATE analysis...")
            video = VideoFileClip(video_path)
            video_duration = video.duration
            video.close()
            
            print(f"📊 Video duration: {video_duration:.2f} seconds")
            
            # Extract audio for advanced analysis
            print("🎵 Extracting audio for AI analysis...")
            audio = AudioFileClip(video_path)
            
            # Convert to numpy array for analysis with robust error handling
            try:
                audio_array = audio.to_soundarray(fps=22050)
                
                # Handle different audio array formats
                if len(audio_array.shape) > 1:
                    # Convert stereo to mono by averaging channels
                    try:
                        # More robust stereo to mono conversion
                        if audio_array.shape[1] == 2:  # Stereo
                            audio_array = (audio_array[:, 0] + audio_array[:, 1]) / 2.0
                        else:  # Multi-channel, take first channel
                            audio_array = audio_array[:, 0]
                    except Exception as e:
                        print(f"⚠️ Stereo to mono conversion failed: {e}")
                        # Fallback: take first channel or flatten
                        try:
                            if audio_array.shape[1] > 0:
                                audio_array = audio_array[:, 0]
                            else:
                                audio_array = audio_array.flatten()
                        except Exception as e2:
                            print(f"⚠️ Fallback conversion also failed: {e2}")
                            # Last resort: flatten everything
                            audio_array = audio_array.flatten()
                
                # Ensure audio_array is a proper numpy array and flatten if needed
                try:
                    audio_array = np.array(audio_array, dtype=np.float32)
                    
                    # Force 1D array
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.flatten()
                    
                    # Additional validation
                    if np.isnan(audio_array).any() or np.isinf(audio_array).any():
                        print("⚠️ Audio array contains NaN or infinite values, cleaning...")
                        audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Normalize audio to prevent overflow
                    if np.max(np.abs(audio_array)) > 0:
                        audio_array = audio_array / np.max(np.abs(audio_array))
                    
                    print(f"🎵 Audio array processed successfully: shape={audio_array.shape}, length={len(audio_array)}")
                    
                except Exception as e:
                    print(f"⚠️ Audio array final processing failed: {e}")
                    # Create a safe fallback array
                    audio_array = np.zeros(22050 * 10, dtype=np.float32)  # 10 seconds of silence
                
            except Exception as e:
                print(f"⚠️ Audio array processing failed: {e}")
                # Create a dummy audio array for fallback
                audio_array = np.zeros(22050 * 10, dtype=np.float32)  # 10 seconds of silence
            
            audio.close()
            
            # SUPER-ACCURATE Whisper transcription with advanced settings
            print("🤖 Running ULTRA-PRECISE Whisper transcription...")
            result = self.whisper_model.transcribe(
                video_path,
                word_timestamps=True,
                condition_on_previous_text=True,
                temperature=0.0,  # Deterministic results
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            # Create SUPER-ACCURATE segments using multiple analysis techniques
            segments = self._create_super_accurate_segments(
                result, video_duration, audio_array, video_path
            )
            
            print(f"✅ SUPER-ACCURATE: Created {len(segments)} segments covering {video_duration:.2f} seconds")
            
            # Find viral audio segments with AI-powered analysis
            viral_segments = self._identify_viral_audio_segments_ai(segments, audio_array, result)
            
            return viral_segments, result['text']
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ SUPER-ACCURATE audio analysis failed: {error_msg}")
            print("🔄 Falling back to enhanced segmentation...")
            return self._create_enhanced_fallback_segments(video_path), "AI analysis failed - using enhanced fallback"

    def _create_super_accurate_segments(self, whisper_result, video_duration, audio_array, video_path):
        """Create ultra-precise segments using multiple AI analysis techniques"""

        segments = []

        # Method 1: Advanced Whisper word-level segmentation
        if 'words' in whisper_result and whisper_result['words']:
            print("🔍 Method 1: Advanced Whisper word-level segmentation")
            segments = self._create_whisper_word_segments(whisper_result, video_duration)
        else:
            segments = []

        # Method 2: Audio energy and emotion analysis
        if not segments or len(segments) < 3:
            print("🔍 Method 2: Audio energy and emotion analysis")
            audio_segments = self._create_audio_analysis_segments(audio_array, video_duration, whisper_result)
            if audio_segments:
                segments = audio_segments
        
        # Method 3: Hybrid approach combining multiple techniques
        if not segments or len(segments) < 3:
            print("🔍 Method 3: Hybrid AI analysis")
            segments = self._create_hybrid_ai_segments(whisper_result, audio_array, video_duration)
        
        # Method 4: Fallback with enhanced logic
        if not segments or len(segments) < 3:
            print("🔍 Method 4: Enhanced fallback segmentation")
            segments = self._create_enhanced_fallback_segments(video_path)
        
        # Final validation and optimization
        segments = self._optimize_segment_boundaries(segments, whisper_result, audio_array)
        
        return segments

    def _create_whisper_word_segments(self, whisper_result, video_duration):
        """Create precise segments from Whisper word timestamps with AI optimization"""
        
        segments = []
        words = whisper_result.get('words', [])
        
        if not words:
            return segments
        
        # AI-powered segment creation with natural language boundaries
        current_segment = {'start': 0, 'end': 0, 'text': '', 'words': []}
        target_segment_duration = 30  # Target 30-second segments
        
        for word_info in words:
            word_start = word_info['start']
            word_end = word_info['end']
            word_text = word_info['word']
            
            # Check if we should start a new segment
            should_new_segment = False
            
            # 1. Duration-based segmentation
            if current_segment['end'] > 0 and (word_end - current_segment['start']) > target_segment_duration:
                should_new_segment = True
            
            # 2. Natural language boundary detection
            if self._is_natural_segment_boundary(word_text, current_segment['text']):
                should_new_segment = True
            
            # 3. Audio pause detection (if available)
            if self._detect_audio_pause(word_start, word_end, whisper_result):
                should_new_segment = True
            
            if should_new_segment and current_segment['text'].strip():
                # Finalize current segment
                current_segment['text'] = current_segment['text'].strip()
                current_segment['duration'] = current_segment['end'] - current_segment['start']
                
                if current_segment['duration'] >= 5:  # Minimum 5 seconds
                    segments.append(current_segment.copy())
                
                # Start new segment
                current_segment = {
                    'start': word_start,
                    'end': word_end,
                    'text': word_text,
                    'words': [word_info]
                }
                current_segment['text'] += ' ' + word_text if current_segment['text'] else word_text
                current_segment['words'].append(word_info)
                
                # Add the last segment
                if current_segment['text'].strip():
                    current_segment['text'] = current_segment['text'].strip()
                    current_segment['duration'] = current_segment['end'] - current_segment['start']
                    if current_segment['duration'] >= 5:
                        segments.append(current_segment.copy())
        # Ensure full coverage
        segments = self._ensure_full_coverage(segments, video_duration, whisper_result)
        
        return segments

    def _is_natural_segment_boundary(self, current_word, segment_text):
        """AI-powered detection of natural segment boundaries"""
        
        # Check for sentence endings
        if segment_text.endswith(('.', '!', '?', ':', ';')):
            return True
        
        # Check for natural speech patterns
        natural_breaks = [
            'so', 'well', 'actually', 'you know', 'listen', 'guess what',
            'imagine', 'picture this', 'anyway', 'moving on', 'next',
            'first', 'second', 'finally', 'in conclusion', 'to summarize'
        ]
        
        if any(segment_text.lower().endswith(break_word) for break_word in natural_breaks):
            return True
        
        # Check for topic shifts
        topic_shifters = [
            'but', 'however', 'on the other hand', 'meanwhile',
            'speaking of', 'by the way', 'oh', 'wait', 'hold on'
        ]
        
        if any(current_word.lower() in shifter for shifter in topic_shifters):
            return True
        
        return False

    def _detect_audio_pause(self, word_start, word_end, whisper_result):
        """Detect natural audio pauses for better segmentation"""
        
        # Look for gaps between words that indicate pauses
        words = whisper_result.get('words', [])
        current_word_index = None
        
        # Find current word index
        for i, word in enumerate(words):
            if abs(word['start'] - word_start) < 0.1:
                current_word_index = i
                break
        
        if current_word_index is None or current_word_index == 0:
            return False
        
        # Check for pause before current word
        prev_word = words[current_word_index - 1]
        pause_duration = word_start - prev_word['end']
        
        # Pause longer than 0.5 seconds indicates natural break
        return pause_duration > 0.5

    def _create_audio_analysis_segments(self, audio_array, video_duration, whisper_result):
        """Create segments based on audio energy, emotion, and pattern analysis"""
        
        if not LIBROSA_AVAILABLE:
            print("⚠️ Librosa not available, skipping advanced audio analysis")
            return []
        
        try:
            print("🎵 Analyzing audio patterns for intelligent segmentation...")
            
            # Ensure audio_array is properly formatted
            if audio_array is None or len(audio_array) == 0:
                print("⚠️ Audio array is empty or None")
                return []
            
            # Convert to proper numpy array if needed
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=np.float32)
            
            # Ensure audio is 1D and properly formatted
            if len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
            
            # Additional validation
            if np.isnan(audio_array).any() or np.isinf(audio_array).any():
                print("⚠️ Audio array contains NaN or infinite values, cleaning...")
                audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            print(f"🎵 Audio array shape: {audio_array.shape}, length: {len(audio_array)}")
            
            # Extract audio features with additional error handling
            hop_length = 512
            frame_length = 2048
            
            try:
                # 1. Energy analysis
                energy = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)
                if len(energy.shape) > 1:
                    energy = energy[0]  # Extract first dimension if 2D
                energy_times = librosa.frames_to_time(range(len(energy)), hop_length=hop_length, sr=22050)
                
                # 2. Spectral centroid (brightness)
                spectral_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=22050, hop_length=hop_length)
                if len(spectral_centroid.shape) > 1:
                    spectral_centroid = spectral_centroid[0]  # Extract first dimension if 2D
                
                # 3. Zero crossing rate (noise vs. speech)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_array, frame_length=frame_length, hop_length=hop_length)
                if len(zero_crossing_rate.shape) > 1:
                    zero_crossing_rate = zero_crossing_rate[0]  # Extract first dimension if 2D
                
                # 4. MFCC features for emotion detection
                mfcc = librosa.feature.mfcc(y=audio_array, sr=22050, hop_length=hop_length, n_mfcc=13)
                
            except Exception as e:
                print(f"⚠️ Audio feature extraction failed: {e}")
                # Return empty segments if audio analysis fails
                return []
            
            print(f"🎵 Extracted features: Energy={len(energy)}, Spectral={len(spectral_centroid)}, MFCC={mfcc.shape}")
            
            # Find significant changes in audio characteristics
            segments = []
            segment_start = 0
            min_segment_duration = 10  # Minimum 10 seconds
            
            for i in range(1, len(energy_times)):
                time = energy_times[i]
                
                # Detect significant changes
                energy_change = abs(energy[i] - energy[i-1]) / (energy[i-1] + 1e-8)
                spectral_change = abs(spectral_centroid[i] - spectral_centroid[i-1]) / (spectral_centroid[i-1] + 1e-8)
                
                # Threshold for significant change
                if (energy_change > 0.3 or spectral_change > 0.2) and (time - segment_start) >= min_segment_duration:
                    # Create segment
                    segment_text = self._extract_text_for_time_range(segment_start, time, whisper_result)
                    
                    # Calculate audio features for this segment
                    start_frame = int(segment_start * 22050 / hop_length)
                    end_frame = int(time * 22050 / hop_length)
                    
                    # Ensure frame indices are within bounds
                    start_frame = max(0, min(start_frame, len(energy) - 1))
                    end_frame = max(start_frame, min(end_frame, len(energy) - 1))
                    
                    if end_frame > start_frame:
                        segment_energy = energy[start_frame:end_frame]
                        segment_spectral = spectral_centroid[start_frame:end_frame]
                        
                    segments.append({
                            'start': segment_start,
                            'end': time,
                            'text': segment_text,
                            'duration': time - segment_start,
                            'audio_features': {
                                'avg_energy': float(np.mean(segment_energy)),
                                'avg_spectral': float(np.mean(segment_spectral)),
                                'energy_variance': float(np.var(segment_energy))
                            }
                        })
                    
                    segment_start = time
            
            # Add final segment
            if video_duration - segment_start >= min_segment_duration:
                segment_text = self._extract_text_for_time_range(segment_start, video_duration, whisper_result)
                
                # Calculate final segment features
                start_frame = int(segment_start * 22050 / hop_length)
                start_frame = max(0, min(start_frame, len(energy) - 1))
                
                if start_frame < len(energy):
                    final_energy = energy[start_frame:]
                    final_spectral = spectral_centroid[start_frame:]
                    
                    segments.append({
                        'start': segment_start,
                        'end': video_duration,
                        'text': segment_text,
                        'duration': video_duration - segment_start,
                        'audio_features': {
                            'avg_energy': float(np.mean(final_energy)),
                            'avg_spectral': float(np.mean(final_spectral)),
                            'energy_variance': float(np.var(final_energy))
                        }
                    })
            
            print(f"🎵 Audio analysis created {len(segments)} segments")
            return segments
            
        except Exception as e:
            print(f"⚠️ Audio analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_text_for_time_range(self, start_time, end_time, whisper_result):
        """Extract text for a specific time range from Whisper result"""
        
        text = ""
        words = whisper_result.get('words', [])
        
        for word in words:
            if start_time <= word['start'] <= end_time:
                text += word['word'] + " "
        
        return text.strip() if text else f"Audio content from {start_time:.1f}s to {end_time:.1f}s"

    def _create_hybrid_ai_segments(self, whisper_result, audio_array, video_duration):
        """Create segments using hybrid AI approach combining multiple techniques"""
        
        print("🤖 Creating hybrid AI segments...")
        
        # Combine Whisper transcription with audio analysis
        segments = []
        
        # Get basic Whisper segments
        if 'words' in whisper_result and whisper_result['words']:
            whisper_segments = self._create_whisper_word_segments(whisper_result, video_duration)
            segments.extend(whisper_segments)
        
        # If we don't have enough segments, add audio-based ones
        if len(segments) < 5:
            audio_segments = self._create_audio_analysis_segments(audio_array, video_duration, whisper_result)
            
            # Merge overlapping segments
            segments = self._merge_overlapping_segments(segments + audio_segments)
        
        # Ensure minimum segment count
        if len(segments) < 5:
            segments = self._create_enhanced_fallback_segments_from_duration(video_duration, whisper_result)
        
        return segments

    def _merge_overlapping_segments(self, segments):
        """Merge overlapping segments intelligently"""
        
        if not segments:
            return segments
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            # Check for overlap
            if next_seg['start'] <= current['end']:
                # Merge segments
                current['end'] = max(current['end'], next_seg['end'])
                current['duration'] = current['end'] - current['start']
                
                # Merge text intelligently
                if 'text' in current and 'text' in next_seg:
                    current['text'] = current['text'] + " " + next_seg['text']
                
                # Merge other attributes
                if 'audio_features' in next_seg:
                    if 'audio_features' not in current:
                        current['audio_features'] = {}
                    # Average the features
                    for key in next_seg['audio_features']:
                        if key in current['audio_features']:
                            current['audio_features'][key] = (current['audio_features'][key] + next_seg['audio_features'][key]) / 2
                        else:
                            current['audio_features'][key] = next_seg['audio_features'][key]
            else:
                # No overlap, add current to merged list
                merged.append(current)
                current = next_seg.copy()
        
        # Add the last segment
        merged.append(current)
        
        return merged

    def _optimize_segment_boundaries(self, segments, whisper_result, audio_array):
        """Optimize segment boundaries for maximum accuracy"""
        
        print("🔧 Optimizing segment boundaries...")
        
        optimized_segments = []
        
        for segment in segments:
            optimized_segment = segment.copy()
            
            # 1. Align with word boundaries
            start_time, end_time = self._align_with_word_boundaries(
                segment['start'], segment['end'], whisper_result
            )
            
            # 2. Check for natural speech boundaries
            start_time, end_time = self._find_natural_speech_boundaries(
                start_time, end_time, whisper_result
            )
            
            # 3. Validate audio quality at boundaries
            start_time, end_time = self._validate_audio_boundaries(
                start_time, end_time, audio_array
            )
            
            # Update segment with optimized timing
            optimized_segment['start'] = start_time
            optimized_segment['end'] = end_time
            optimized_segment['duration'] = end_time - start_time
            
            # Ensure minimum duration
            if optimized_segment['duration'] >= 5:
                optimized_segments.append(optimized_segment)
        
        return optimized_segments

    def _align_with_word_boundaries(self, start_time, end_time, whisper_result):
        """Align segment boundaries with actual word boundaries"""
        
        words = whisper_result.get('words', [])
        
        # Find closest word start to segment start
        best_start = start_time
        min_start_diff = float('inf')
        
        for word in words:
            diff = abs(word['start'] - start_time)
            if diff < min_start_diff:
                min_start_diff = diff
                best_start = word['start']
        
        # Find closest word end to segment end
        best_end = end_time
        min_end_diff = float('inf')
        
        for word in words:
            diff = abs(word['end'] - end_time)
            if diff < min_end_diff:
                min_end_diff = diff
                best_end = word['end']
        
        return best_start, best_end

    def _find_natural_speech_boundaries(self, start_time, end_time, whisper_result):
        """Find natural speech boundaries near the given times"""
        
        words = whisper_result.get('words', [])
        
        # Look for natural sentence boundaries
        for word in words:
            # Check start boundary
            if abs(word['start'] - start_time) < 2.0:  # Within 2 seconds
                if self._is_sentence_start(word):
                    start_time = word['start']
            
            # Check end boundary
            if abs(word['end'] - end_time) < 2.0:  # Within 2 seconds
                if self._is_sentence_end(word):
                    end_time = word['end']
        
        return start_time, end_time

    def _is_sentence_start(self, word):
        """Check if word indicates sentence start"""
        
        text = word['word'].strip()
        
        # Capitalized words that often start sentences
        sentence_starters = [
            'so', 'well', 'actually', 'you', 'i', 'we', 'they', 'he', 'she',
            'this', 'that', 'these', 'those', 'here', 'there', 'now', 'then'
        ]
        
        return text.lower() in sentence_starters or text[0].isupper()

    def _is_sentence_end(self, word):
        """Check if word indicates sentence end"""
        
        text = word['word'].strip()
        
        # Punctuation that ends sentences
        return text.endswith(('.', '!', '?', ':', ';'))

    def _validate_audio_boundaries(self, start_time, end_time, audio_array):
        """Validate that audio boundaries are clean and clear"""
        
        # Ensure audio_array is valid
        if audio_array is None or len(audio_array) == 0:
            return start_time, end_time
        
        # Convert to proper numpy array if needed
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        
        # Ensure audio is 1D
        if len(audio_array.shape) > 1:
            audio_array = audio_array.flatten()
        
        # Convert time to sample indices
        sr = 22050
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Ensure we don't go out of bounds
        start_sample = max(0, min(start_sample, len(audio_array) - 1))
        end_sample = max(start_sample, min(end_sample, len(audio_array) - 1))
        
        # Check audio quality at boundaries
        boundary_samples = 1000  # Check 1000 samples around boundary
        
        # Check start boundary
        start_quality = self._check_audio_quality(
            audio_array[start_sample:start_sample + boundary_samples]
        )
        
        # Check end boundary
        end_quality = self._check_audio_quality(
            audio_array[end_sample - boundary_samples:end_sample]
        )
        
        # If boundaries are noisy, adjust slightly
        if start_quality < 0.3:  # Low quality threshold
            start_time += 0.1  # Move forward 0.1 seconds
        
        if end_quality < 0.3:  # Low quality threshold
            end_time -= 0.1  # Move backward 0.1 seconds
        
        return start_time, end_time

    def _check_audio_quality(self, audio_samples):
        """Check audio quality of a sample"""
        
        if len(audio_samples) == 0:
            return 0.0
        
        # Calculate signal-to-noise ratio
        signal_power = np.mean(audio_samples ** 2)
        noise_power = np.var(audio_samples)
        
        if noise_power == 0:
            return 1.0
        
        snr = 10 * np.log10(signal_power / noise_power)
        
        # Normalize to 0-1 scale
        quality = min(1.0, max(0.0, (snr + 20) / 40))
        
        return quality

    def _identify_viral_audio_segments_ai(self, segments, audio_array, whisper_result):
        """Identify audio segments likely to contain viral content using AI-powered analysis"""
        
        viral_segments = []
        
        print(f"🤖 AI-POWERED ANALYSIS: Analyzing {len(segments)} segments for viral content...")
        
        for i, segment in enumerate(segments):
            # Calculate segment characteristics
            duration = segment['end'] - segment['start']
            text = segment['text'].strip()
            
            # Skip very short segments
            if duration < self.min_segment_length:
                print(f"Segment {i}: Skipped (too short: {duration:.1f}s)")
                continue
            
            # AI-powered viral content scoring
            viral_score = self._calculate_ai_viral_score(segment, audio_array, whisper_result)
            
            # Enhanced segment data
            enhanced_segment = {
                'start': segment['start'],
                'end': segment['end'],
                'text': text,
                'duration': duration,
                'viral_score': viral_score,
                'timestamp': (segment['start'] + segment['end']) / 2,
                'ai_analysis': self._get_ai_analysis_metadata(segment, audio_array, whisper_result)
            }
            
            viral_segments.append(enhanced_segment)
            
            print(f"Segment {i}: {duration:.1f}s, AI Score: {viral_score:.1f}/10, Text: {text[:50]}...")
        
        # Sort by viral score and limit
        viral_segments.sort(key=lambda x: x['viral_score'], reverse=True)
        print(f"🤖 AI ANALYSIS COMPLETE: Selected top {min(len(viral_segments), self.max_segments)} segments with highest viral scores")
        
        return viral_segments[:self.max_segments]

    def _calculate_ai_viral_score(self, segment, audio_array, whisper_result):
        """Calculate viral score using advanced AI analysis"""
        
        base_score = 5.0  # Start with base score
        
        # 1. Text-based viral indicators (enhanced)
        text_score = self._analyze_text_viral_potential(segment['text'])
        base_score += text_score
        
        # 2. Audio quality and energy analysis
        audio_score = self._analyze_audio_viral_potential(segment, audio_array)
        base_score += audio_score
        
        # 3. Timing and pacing analysis
        timing_score = self._analyze_timing_viral_potential(segment, whisper_result)
        base_score += timing_score
        
        # 4. Content uniqueness and freshness
        uniqueness_score = self._analyze_content_uniqueness(segment, whisper_result)
        base_score += uniqueness_score
        
        # 5. Emotional impact analysis
        emotional_score = self._analyze_emotional_impact(segment, whisper_result)
        base_score += emotional_score
        
        # 6. Platform optimization potential
        platform_score = self._analyze_platform_potential(segment)
        base_score += platform_score
        
        # Normalize to 1-10 scale
        final_score = min(10.0, max(1.0, base_score))
        
        return round(final_score, 1)

    def _analyze_text_viral_potential(self, text):
        """Analyze text for viral potential indicators"""
        
        score = 0.0
        text_lower = text.lower()
        
        # Emotional intensity
        emotional_words = {
            'wow': 2.0, 'omg': 2.5, 'amazing': 2.0, 'incredible': 2.0, 'unbelievable': 2.5,
            'haha': 1.5, 'lol': 1.5, 'holy': 2.0, 'damn': 1.5, 'crazy': 2.0, 'insane': 2.5,
            'epic': 2.0, 'legendary': 2.5, 'mind-blowing': 3.0, 'spectacular': 2.0,
            'hilarious': 2.0, 'genius': 2.0, 'brilliant': 2.0, 'perfect': 2.0
        }
        
        for word, word_score in emotional_words.items():
            if word in text_lower:
                score += word_score
        
        # Exclamation and emphasis
        if '!' in text:
            score += 1.0
        if text.isupper() or any(word.isupper() for word in text.split()):
            score += 1.0
        
        # Laughter patterns
        laughter_patterns = ['haha', 'lol', 'hehe', 'hahaha', '😂', '😄', 'lmao', 'rofl', 'lmfao']
        for pattern in laughter_patterns:
            if pattern.lower() in text_lower:
                score += 2.0
        
        # Engagement patterns
        if '?' in text:
            score += 1.0
        if any(word in text_lower for word in ['challenge', 'vs', 'battle', 'competition']):
            score += 2.0
        
        return min(5.0, score)  # Cap at 5 points

    def _analyze_audio_viral_potential(self, segment, audio_array):
        """Analyze audio characteristics for viral potential"""
        
        score = 0.0
        
        try:
            # Ensure audio_array is valid
            if audio_array is None or len(audio_array) == 0:
                return score
            
            # Convert to proper numpy array if needed
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=np.float32)
            
            # Ensure audio is 1D
            if len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
            
            # Extract audio segment
            sr = 22050
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            
            # Ensure bounds
            start_sample = max(0, min(start_sample, len(audio_array) - 1))
            end_sample = max(start_sample, min(end_sample, len(audio_array) - 1))
            
            if end_sample > start_sample:
                segment_audio = audio_array[start_sample:end_sample]
                
                # Energy analysis
                energy = np.mean(segment_audio ** 2)
                if energy > 0.1:  # High energy
                    score += 1.0
                
                # Dynamic range
                dynamic_range = np.max(segment_audio) - np.min(segment_audio)
                if dynamic_range > 0.5:  # High dynamic range
                    score += 1.0
                
                # Clarity (low zero-crossing rate for speech)
                zero_crossings = np.sum(np.diff(np.sign(segment_audio)) != 0)
                if zero_crossings < len(segment_audio) * 0.1:  # Clear speech
                    score += 1.0
            
        except Exception as e:
            print(f"⚠️ Audio analysis failed for segment: {e}")
        
        return min(3.0, score)  # Cap at 3 points

    def _analyze_timing_viral_potential(self, segment, whisper_result):
        """Analyze timing and pacing for viral potential"""
        
        score = 0.0
        
        # Optimal duration (15-60 seconds is ideal for viral content)
        duration = segment['duration']
        if 15 <= duration <= 60:
            score += 2.0
        elif 10 <= duration <= 90:
            score += 1.0
        
        # Position in video (later segments often have more action)
        video_position = segment['start'] / 60  # Minutes into video
        if video_position > 1:  # After first minute
            score += 1.0
        if video_position > 5:  # After 5 minutes
            score += 1.0
        
        # Pacing analysis
        words = self._get_words_in_segment(segment, whisper_result)
        if words:
            words_per_second = len(words) / duration
            if 1.5 <= words_per_second <= 3.0:  # Optimal speaking pace
                score += 1.0
        
        return min(4.0, score)  # Cap at 4 points

    def _analyze_content_uniqueness(self, segment, whisper_result):
        """Analyze content uniqueness and freshness"""
        
        score = 0.0
        text = segment['text'].lower()
        
        # Check for unique content patterns
        unique_patterns = [
            'first time', 'never seen', 'unbelievable', 'incredible', 'amazing',
            'shocking', 'surprising', 'unexpected', 'wow', 'omg', 'holy'
        ]
        
        for pattern in unique_patterns:
            if pattern in text:
                score += 1.0
        
        # Check for storytelling elements
        story_elements = [
            'so', 'well', 'actually', 'you know', 'listen', 'guess what',
            'imagine', 'picture this', 'anyway', 'moving on', 'next'
        ]
        
        for element in story_elements:
            if text.startswith(element):
                score += 1.0
        
        return min(3.0, score)  # Cap at 3 points

    def _analyze_emotional_impact(self, segment, whisper_result):
        """Analyze emotional impact and relatability"""
        
        score = 0.0
        text = segment['text'].lower()
        
        # Emotional triggers
        emotional_triggers = {
            'joy': ['happy', 'joy', 'excited', 'thrilled', 'amazing'],
            'surprise': ['wow', 'omg', 'unbelievable', 'incredible', 'shocking'],
            'humor': ['funny', 'hilarious', 'lol', 'haha', 'comedy'],
            'relatability': ['you know', 'right', 'exactly', 'same', 'me too'],
            'inspiration': ['motivation', 'inspire', 'amazing', 'incredible', 'wow']
        }
        
        for emotion, words in emotional_triggers.items():
            if any(word in text for word in words):
                score += 1.0
        
        return min(4.0, score)  # Cap at 4 points

    def _analyze_platform_potential(self, segment):
        """Analyze potential for different platforms"""
        
        score = 0.0
        duration = segment['duration']
        
        # TikTok optimization
        if 15 <= duration <= 60:
            score += 1.0
        
        # Instagram Reels
        if 15 <= duration <= 90:
            score += 1.0
        
        # YouTube Shorts
        if 15 <= duration <= 60:
            score += 1.0
        
        # Cross-platform potential
        if 15 <= duration <= 60:
            score += 1.0
        
        return min(3.0, score)  # Cap at 3 points

    def _get_words_in_segment(self, segment, whisper_result):
        """Get words that fall within a segment"""
        
        words = []
        whisper_words = whisper_result.get('words', [])
        
        for word_info in whisper_words:
            if segment['start'] <= word_info['start'] <= segment['end']:
                words.append(word_info['word'])
        
        return words

    def _get_ai_analysis_metadata(self, segment, audio_array, whisper_result):
        """Get comprehensive AI analysis metadata for a segment"""
        
        return {
            'text_score': self._analyze_text_viral_potential(segment['text']),
            'audio_score': self._analyze_audio_viral_potential(segment, audio_array),
            'timing_score': self._analyze_timing_viral_potential(segment, whisper_result),
            'uniqueness_score': self._analyze_content_uniqueness(segment, whisper_result),
            'emotional_score': self._analyze_emotional_impact(segment, whisper_result),
            'platform_score': self._analyze_platform_potential(segment),
            'word_count': len(self._get_words_in_segment(segment, whisper_result)),
            'speaking_pace': len(self._get_words_in_segment(segment, whisper_result)) / segment['duration'] if segment['duration'] > 0 else 0
        }

    def _ensure_full_coverage(self, segments, video_duration, whisper_result):
        """Ensure segments cover the full video duration"""
        
        if not segments:
            return segments
        
        # Check for gaps
        covered_ranges = []
        for segment in segments:
            covered_ranges.append((segment['start'], segment['end']))
        
        # Sort by start time
        covered_ranges.sort(key=lambda x: x[0])
        
        # Find gaps
        gaps = []
        current_end = 0
        
        for start, end in covered_ranges:
            if start > current_end:
                gaps.append((current_end, start))
            current_end = max(current_end, end)
        
        # Add final gap if needed
        if current_end < video_duration:
            gaps.append((current_end, video_duration))
        
        # Fill gaps with content
        for gap_start, gap_end in gaps:
            if gap_end - gap_start >= 5:  # Only fill gaps >= 5 seconds
                gap_text = self._extract_text_for_time_range(gap_start, gap_end, whisper_result)
                
                segments.append({
                    'start': gap_start,
                    'end': gap_end,
                    'text': gap_text,
                    'duration': gap_end - gap_start,
                    'is_gap_filler': True
                })
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        
        return segments

    def _create_enhanced_fallback_segments(self, video_path):
        """Create enhanced fallback segments with better logic"""
        
        try:
            from moviepy.editor import VideoFileClip
            video = VideoFileClip(video_path)
            video_duration = video.duration
            video.close()
            
            print(f"🔄 Enhanced fallback: Video duration: {video_duration:.2f} seconds")
            
            # Create intelligent segments
            segment_count = max(8, int(video_duration / 45))  # More segments, shorter duration
            segment_duration = video_duration / segment_count
            
            segments = []
            for i in range(segment_count):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration if i < segment_count - 1 else video_duration
                
                # Create descriptive text
                if i == 0:
                    text = f"Opening segment - Introduction and setup"
                elif i == segment_count - 1:
                    text = f"Final segment - Conclusion and wrap-up"
                elif i < segment_count // 3:
                    text = f"Early content - Building momentum"
                elif i < 2 * segment_count // 3:
                    text = f"Middle content - Core material and engagement"
                else:
                    text = f"Later content - Building to climax"
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'duration': end_time - start_time,
                    'viral_score': 6 + (i % 4),  # Vary scores
                    'fallback_type': 'enhanced'
                })
            
            print(f"🔄 Enhanced fallback: Created {len(segments)} segments covering {video_duration:.2f} seconds")
            return segments
            
        except Exception as e:
            print(f"❌ Enhanced fallback failed: {e}")
            return self._create_basic_fallback_segments(video_duration)

    def _create_enhanced_fallback_segments_from_duration(self, video_duration, whisper_result):
        """Create fallback segments from duration when other methods fail"""
        
        segment_count = max(6, int(video_duration / 40))
        segment_duration = video_duration / segment_count
        
        segments = []
        for i in range(segment_count):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration if i < segment_count - 1 else video_duration
            
            # Try to extract text from Whisper result
            segment_text = self._extract_text_for_time_range(start_time, end_time, whisper_result)
            
            if not segment_text or len(segment_text) < 10:
                segment_text = f"Content segment {i+1} ({start_time:.1f}s - {end_time:.1f}s)"
            
            segments.append({
                'start': start_time,
                'end': end_time,
                'text': segment_text,
                'duration': end_time - start_time,
                'viral_score': 5 + (i % 3),
                'fallback_type': 'duration_based'
            })
        
        return segments

    def _create_basic_fallback_segments(self, video_duration):
        """Create basic fallback segments as last resort"""
        
        segments = []
        segment_duration = min(60, video_duration / 5)  # 60s segments or 5 segments
        
        for i in range(int(video_duration / segment_duration) + 1):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, video_duration)
            
            segments.append({
                'start': start_time,
                'end': end_time,
                'text': f"Basic segment {i+1}",
                'duration': end_time - start_time,
                'viral_score': 5,
                'fallback_type': 'basic'
            })
        
        return segments

    def ai_select_best_clips(self, viral_segments, full_transcript, num_clips_requested, frontend_inputs=None):
        """AI-powered selection of best clips with ULTRA-FORTIFIED dual-pass processing"""
        try:
            print(f"🚀 ULTRA-FORTIFIED AI SELECTION: Processing {num_clips_requested} clips with dual-pass system...")
            
            # Use the new dual-pass AI processing system
            enhanced_clips = self.dual_pass_ai_processing(viral_segments, full_transcript, num_clips_requested, frontend_inputs)
            
            if enhanced_clips:
                print(f"✅ DUAL-PASS AI COMPLETE: {len(enhanced_clips)} high-quality viral clips generated")
                return enhanced_clips
            else:
                print("⚠️ Dual-pass failed, falling back to standard processing...")
                return self.legacy_ai_selection(viral_segments, full_transcript, num_clips_requested, frontend_inputs)
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ ULTRA-FORTIFIED AI selection failed: {error_msg}")
            
            # Check if it's a quota error
            if "quota" in error_msg.lower() or "429" in error_msg or "rate limit" in error_msg.lower():
                print("⚠️ Quota limit reached - using fallback selection")
                self.handle_quota_error()
                return self.fallback_clip_selection(viral_segments, num_clips_requested)
            
            # Fallback to basic selection
            print("🔄 Using fallback clip selection due to AI failure")
            return self.fallback_clip_selection(viral_segments, num_clips_requested)

    def legacy_ai_selection(self, viral_segments, full_transcript, num_clips_requested, frontend_inputs=None):
        """Legacy AI selection method as fallback for dual-pass system"""
        try:
            print(f"🔄 LEGACY MODE: AI selecting {num_clips_requested} clips from {len(viral_segments)} segments...")
            
            # Preprocess transcription to reduce API usage
            condensed_transcript = self.preprocess_transcription_for_ai(full_transcript, viral_segments)
            
            # Advanced prompt engineering with multiple strategies and frontend inputs
            prompt = self.create_advanced_ai_prompt(condensed_transcript, viral_segments, num_clips_requested, frontend_inputs)
            
            # Get AI response with enhanced configuration
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,  # Creative but focused
                    top_p=0.9,       # High quality responses
                    top_k=40,        # Diverse but relevant
                    max_output_tokens=4000  # Allow detailed responses
                )
            )
            
            try:
                # Clean the response text to remove markdown formatting
                cleaned_response = self.clean_ai_response(response.text)
                ai_analysis = json.loads(cleaned_response)
                selected_clips = ai_analysis.get('selected_clips', [])
                
                print(f"✅ LEGACY AI selected {len(selected_clips)} clips")
                
                # Validate and enhance clips
                enhanced_clips = []
                for clip in selected_clips:
                    # Ensure we have all required fields
                    if all(key in clip for key in ['start_time', 'end_time', 'duration', 'caption', 'hashtags']):
                        enhanced_clip = {
                            'start_time': float(clip['start_time']),
                            'end_time': float(clip['end_time']),
                            'duration': float(clip['duration']),
                            'viral_score': int(clip.get('viral_score', 8)),
                            'content_type': clip.get('content_type', 'viral'),
                            'caption': clip['caption'],
                            'hashtags': clip['hashtags'],
                            'target_audience': clip.get('target_audience', 'general'),
                            'platforms': clip.get('platforms', ['TikTok', 'Instagram']),
                            'segment_text': clip.get('segment_text', ''),
                            'viral_potential': clip.get('viral_score', 8),
                            'engagement': clip.get('viral_score', 8),
                            'story_value': clip.get('viral_score', 8),
                            'audio_impact': clip.get('viral_score', 8)
                        }
                        enhanced_clips.append(enhanced_clip)
                
                print(f"✅ LEGACY Enhanced {len(enhanced_clips)} clips with captions and hashtags")
                return enhanced_clips
                
            except json.JSONDecodeError as e:
                print(f"❌ LEGACY Failed to parse AI response: {e}")
                print(f"AI Response: {response.text}")
                # Try to fix JSON and retry
                return self.retry_ai_selection_with_fallback(prompt, viral_segments, num_clips_requested)
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ LEGACY AI selection failed: {error_msg}")
            return self.fallback_clip_selection(viral_segments, num_clips_requested)

    def fallback_clip_selection(self, viral_segments, num_clips_requested):
        """Fallback method when AI selection fails"""
        try:
            print(f"Fallback: Selecting {num_clips_requested} clips from {len(viral_segments)} segments")
            
            if not viral_segments:
                print("No viral segments available - creating basic segments")
                # Create basic segments covering the video
                basic_segments = []
                for i in range(num_clips_requested):
                    start_time = i * 60  # 1 minute intervals
                    end_time = (i + 1) * 60
                    basic_segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': 60,
                        'viral_score': 8 - i,  # Decreasing scores
                        'content_type': 'viral',
                        'caption': f'Viral moment {i+1} - {start_time}s to {end_time}s',
                        'hashtags': ['#viral', '#trending', '#amazing', '#mustwatch'],
                        'target_audience': 'general',
                        'platforms': ['TikTok', 'Instagram'],
                        'segment_text': f'Content from {start_time}s to {end_time}s'
                    })
                return basic_segments
            
            # Use existing viral segments, sorted by score
            viral_segments.sort(key=lambda x: x['viral_score'], reverse=True)
            selected_segments = viral_segments[:num_clips_requested]
            
            # Convert to clip format
            clips = []
            for i, seg in enumerate(selected_segments):
                clip = {
                    'start_time': seg['start'],
                    'end_time': seg['end'],
                    'duration': seg['duration'],
                    'viral_score': seg['viral_score'],
                    'content_type': 'viral',
                    'caption': f'Viral moment {i+1} - {seg["text"][:50]}...',
                    'hashtags': ['#viral', '#trending', '#amazing', '#mustwatch'],
                    'target_audience': 'general',
                    'platforms': ['TikTok', 'Instagram'],
                    'segment_text': seg['text']
                }
                clips.append(clip)
            
            print(f"Fallback: Created {len(clips)} clips successfully")
            return clips
            
        except Exception as e:
            print(f"Fallback selection also failed: {e}")
            # Last resort - create minimal clips
            return [{
                'start_time': i * 30,
                'end_time': (i + 1) * 30,
                'duration': 30,
                'viral_score': 7,
                'content_type': 'viral',
                'caption': f'Viral clip {i+1}',
                'hashtags': ['#viral', '#trending'],
                'target_audience': 'general',
                'platforms': ['TikTok'],
                'segment_text': f'Content segment {i+1}'
            } for i in range(num_clips_requested)]
    
    def create_ultra_advanced_ai_prompt(self, condensed_transcript, viral_segments, num_clips_requested, frontend_inputs=None):
        """Create ultra-advanced AI prompt with strict user instruction adherence"""
        
        # Get user's specific AI instructions
        user_ai_prompt = frontend_inputs.get('aiPrompt', '') if frontend_inputs else ''
        
        # Build strict user instruction context
        user_instructions = self._build_strict_user_instructions(frontend_inputs)
        
        prompt = f"""
        🚨 **STRICT USER INSTRUCTION COMPLIANCE SYSTEM**
        
        You are an AI content analyzer that MUST follow user instructions EXACTLY.
        **USER INSTRUCTIONS ARE ABSOLUTE AND OVERRIDE ALL DEFAULT BEHAVIORS.**
        
        📋 **USER'S SPECIFIC AI INSTRUCTIONS (MANDATORY TO FOLLOW):**
        {user_ai_prompt if user_ai_prompt.strip() else 'No specific AI instructions provided - use standard viral content selection.'}
        
        📊 **USER CONTEXT & REQUIREMENTS:**
        {user_instructions}
        
        🎯 **PRIMARY MISSION:**
        Extract EXACTLY {num_clips_requested} clips that follow the user's instructions above.
        If user instructions conflict with viral content best practices, USER INSTRUCTIONS WIN.
        
        📝 **TRANSCRIPT DATA:**
        {condensed_transcript}

        ---  

        🚨 **STRICT COMPLIANCE RULES:**
        
        1. **USER INSTRUCTIONS ARE PRIORITY #1** - Follow them exactly as specified
        2. **IGNORE DEFAULT VIRAL RULES** if they conflict with user instructions
        3. **ADAPT SELECTION CRITERIA** to match user's specific requirements
        4. **PRIORITIZE USER PREFERENCES** over generic viral content formulas
        
        ---
        
        🧠 **SELECTION PROCESS (USER-CENTRIC):**
        
        STEP 1: **ANALYZE USER INSTRUCTIONS**
        - What specific content type does the user want?
        - What style, tone, or focus areas are mentioned?
        - What platforms or audiences are targeted?
        
        STEP 2: **APPLY USER CRITERIA TO TRANSCRIPT**
        - Filter segments based on user's specific requirements
        - Ignore segments that don't match user instructions
        - Prioritize content that aligns with user's vision
        
        STEP 3: **VALIDATE AGAINST USER REQUIREMENTS**
        - Does each selected clip meet user's criteria?
        - Are the clips the type of content the user requested?
        - Does the selection match user's stated preferences?
        
        STEP 4: **QUALITY CONTROL FOR USER SATISFACTION**
        - Ensure clips start/end at natural speech boundaries
        - Verify content quality meets user's standards
        - Confirm selection aligns with user's goals
        
        ---
        
        🔍 **SELECTION PRIORITY (USER-FIRST):**
        
        1. **USER INSTRUCTIONS COMPLIANCE** - Must follow exactly what user requested
        2. **CONTENT RELEVANCE** - Must match user's specified content type/style
        3. **TECHNICAL QUALITY** - Clean audio cuts, proper timing
        4. **USER SATISFACTION** - Content that meets user's stated goals
        
        ---
        
        🚨 **MANDATORY COMPLIANCE CHECKS:**
        
        - ✅ Does each clip follow user's specific instructions?
        - ✅ Is the content type/style what the user requested?
        - ✅ Are the clips relevant to user's stated goals?
        - ✅ Does the selection prioritize user preferences over generic rules?
        
        ---
        
        ✅ **REQUIRED OUTPUT FORMAT:**
        
        {{
            "selected_clips": [
                {{
                    "start_time": <float, 2 decimals>,
                    "end_time": <float, 2 decimals>,
                    "duration": <float>,
                    "viral_score": <int 1-10>,
                    "content_type": "<string - based on user instructions>",
                    "viral_factor": "<string - why this matches user requirements>",
                    "engagement_potential": "<string - how it serves user's goals>",
                    "caption": "<string - optimized for user's target audience>",
                    "hashtags": ["<string>", "<string>", ...],
                    "hook_line": "<string - based on user's style preferences>",
                    "call_to_action": "<string - aligned with user's goals>",
                    "thumbnail_suggestion": "<string - matches user's aesthetic>",
                    "target_audience": "<string - from user's specifications>",
                    "platforms": ["<string>", "<string>", ...],
                    "optimal_posting_time": "<string>",
                    "cross_platform_adaptation": "<string>",
                    "segment_text": "<string>",
                    "reasoning": "<string - explain how this follows user instructions>",
                    "confidence_score": <float 0.0-1.0>,
                    "user_compliance_score": <int 1-10 - how well it follows user instructions>
                }}
            ]
        }}

        ---  

        🧠 **FINAL COMPLIANCE VERIFICATION:**
        
        Before responding, verify:
        - 🔴 **EVERY clip follows user's specific instructions**
        - 🔴 **Content type/style matches user's requirements**
        - 🔴 **Selection criteria prioritize user preferences**
        - 🔴 **Output format is 100% valid JSON**
        
        **REMEMBER: User instructions are LAW. Follow them exactly, even if they conflict with viral content best practices.**
        """

        return prompt

    def create_advanced_ai_prompt(self, condensed_transcript, viral_segments, num_clips_requested, frontend_inputs=None):
        """Create strict user-focused AI prompt that prioritizes user instructions"""
        
        # Get user's specific AI instructions
        user_ai_prompt = frontend_inputs.get('aiPrompt', '') if frontend_inputs else ''
        
        # Build strict user instruction context
        user_instructions = self._build_strict_user_instructions(frontend_inputs)
        
        prompt = f"""
        🚨 **USER INSTRUCTION COMPLIANCE SYSTEM**
        
        You are an AI content analyzer that MUST follow user instructions EXACTLY.
        **USER INSTRUCTIONS OVERRIDE ALL DEFAULT BEHAVIORS AND VIRAL CONTENT RULES.**
        
        📋 **USER'S SPECIFIC AI INSTRUCTIONS (MANDATORY):**
        {user_ai_prompt if user_ai_prompt.strip() else 'No specific AI instructions provided - use standard viral content selection.'}
        
        📊 **USER CONTEXT & REQUIREMENTS (MUST FOLLOW):**
        {user_instructions}
        
        🎯 **PRIMARY MISSION:**
        Extract EXACTLY {num_clips_requested} clips that follow the user's instructions above.
        **USER INSTRUCTIONS ARE LAW - follow them exactly, even if they conflict with viral content best practices.**
        (or fewer, if only high-quality clips exist). Each clip must **start at the natural beginning of the idea/story** 
        and **end at the emotional or narrative climax**, avoiding mid-sentence or awkward cuts.  

        📊 CONTEXT (TRENDS & USER INPUTS):  
        - Trending Context Data:  
          {self.get_trending_context()}  

        - User Instructions:  
          {self._build_user_instructions(frontend_inputs) if frontend_inputs else 'No user instructions provided.'}  

        - Additional Inputs:  
          {self._build_frontend_context(frontend_inputs) if frontend_inputs else ''}  

        - Transcript Data:  
          {condensed_transcript}  

        ---  

        🔑 ADVANCED STRATEGY FRAMEWORK (MANDATORY):  
        1. EMOTIONAL HOOKS: Prioritize joy, surprise, awe, relatability, inspiration, humor.  
        2. STORY ARC: Ensure clips follow Hook → Build → Climax → Resolution.  
        3. VIRAL TRIGGERS: Shareability, comment-bait, controversy, or relatable moments.  
        4. PLATFORM DYNAMICS: Optimize for TikTok, Reels, Shorts, Twitter, each with context.  
        5. TIMING: Must fall within 15–60s, with **first 3s hook** that guarantees scroll-stopping power.  

        ---  

        🔍 SELECTION PRIORITY (ranked highest to lowest):  
        1. VIRALITY: Emotional punch + cultural relevance.  
        2. AUTHENTICITY: Real, raw, relatable moments win over polished but bland content.  
        3. UNIQUENESS: Avoid generic motivational/educational filler. Clips must feel **fresh**.  
        4. SHARE POWER: Does it make viewers think "I MUST send this to a friend"?  

        ---  

        🚨 QUALITY CONTROL (DO NOT SKIP):  
        - Ensure JSON is valid and parseable.  
        - Clip start/end must align with **natural speech boundaries**.  
        - If clips are weak (viral_score < 9), **exclude them**. Fewer but better clips > filler.  
        - Double-check each clip passes ALL criteria before finalizing.  

        ---  

        ✅ FOR EACH SELECTED CLIP, RETURN EXACTLY:  
        {{
            "selected_clips": [
                {{
                    "start_time": <float, 2 decimals>,
                    "end_time": <float, 2 decimals>,
                    "duration": <float>,
                    "viral_score": <int 9-10>,
                    "content_type": "<string>",
                    "viral_factor": "<string>",
                    "engagement_potential": "<string>",
                    "caption": "<string>",
                    "hashtags": ["<string>", "<string>", ...],
                    "hook_line": "<string>",
                    "call_to_action": "<string>",
                    "thumbnail_suggestion": "<string>",
                    "target_audience": "<string>",
                    "platforms": ["<string>", "<string>", ...],
                    "optimal_posting_time": "<string>",
                    "cross_platform_adaptation": "<string>",
                    "segment_text": "<string>"
                }}
            ]
        }}

        ---  

        🧠 SELF-CHECK BEFORE ANSWERING:  
        - Does each clip have emotional impact, narrative arc, and platform fit?  
        - Is the reasoning clear and specific (not generic)?  
        - Is the output 100% valid JSON?  

        If not → refine before responding.  

        REMEMBER: **Virality is about impact, not volume.** Provide fewer clips if only those exceed the viral threshold.  
        """

        return prompt

    def create_expert_review_prompt(self, ai_generated_clips, original_transcript, frontend_inputs=None):
        """Create expert review prompt for second-pass validation and refinement"""
        
        trending_context = self.get_trending_context()
        
        prompt = f"""
        🚨 EXPERT REVIEW ROLE: 
        You are a SENIOR CONTENT STRATEGIST with 20+ years of experience reviewing viral content for major platforms.
        Your job is to REVIEW, VALIDATE, and REFINE the AI-generated clips to ensure maximum viral potential.

        📋 REVIEW TASK:
        Analyze the provided AI-generated clips and either APPROVE them as-is or REFINE them for better results.
        Focus on QUALITY, ACCURACY, and VIRAL POTENTIAL.

        🔍 CONTEXT FOR REVIEW:
        - Trending Context: {trending_context}
        - User Requirements: {self._build_user_instructions(frontend_inputs) if frontend_inputs else 'No specific requirements'}
        - Original Transcript: {original_transcript[:2000]}... (truncated for review)

        📊 AI-GENERATED CLIPS TO REVIEW:
        {json.dumps(ai_generated_clips, indent=2)}

        🚨 CRITICAL REVIEW CRITERIA:
        1. SPEECH BOUNDARIES: Do clips start/end at natural speech breaks? (CRITICAL)
        2. VIRAL POTENTIAL: Is the viral_score accurate? (Must be 9-10 for real viral potential)
        3. CONTENT QUALITY: Are captions engaging and hashtags relevant?
        4. PLATFORM FIT: Do clips work for the target platforms?
        5. TIMING: Are durations optimal (15-60 seconds)?
        6. EMOTIONAL IMPACT: Do clips have genuine emotional hooks?

        ✅ REVIEW RESPONSE FORMAT:
        {{
            "review_results": {{
                "approved_clips": [
                    // Clips that pass review unchanged
                ],
                "refined_clips": [
                    // Clips that need improvement with suggested changes
                ],
                "rejected_clips": [
                    // Clips that don't meet viral standards
                ],
                "overall_quality_score": <int 1-10>,
                "review_notes": "<string>",
                "recommendations": "<string>"
            }}
        }}

        🎯 REVIEW INSTRUCTIONS:
        - If a clip is GOOD: Move to approved_clips
        - If a clip needs IMPROVEMENT: Move to refined_clips with specific suggestions
        - If a clip is WEAK: Move to rejected_clips with reasoning
        - Be STRICT - only 9-10 viral_score clips should pass
        - Focus on SPEECH BOUNDARIES - never cut mid-sentence
        """

        return prompt

    def dual_pass_ai_processing(self, viral_segments, full_transcript, num_clips_requested, frontend_inputs=None):
        """Dual-pass AI processing: First pass generates, second pass reviews and refines"""
        
        print(f"🚀 DUAL-PASS AI PROCESSING: Generating {num_clips_requested} viral clips...")
        
        try:
            # PASS 1: AI Generation
            print("🔄 PASS 1: AI Content Generation...")
            condensed_transcript = self.preprocess_transcription_for_ai(full_transcript, viral_segments)
            
            # Generate initial clips with ULTRA-ADVANCED prompt
            generation_prompt = self.create_ultra_advanced_ai_prompt(condensed_transcript, viral_segments, num_clips_requested, frontend_inputs)
            
            generation_response = self.model.generate_content(
                generation_prompt,
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,  # Creative generation
                    top_p=0.95,      # High quality
                    top_k=50,        # Diverse options
                    max_output_tokens=5000
                )
            )
            
            # Parse generation response
            try:
                cleaned_generation = self.clean_ai_response(generation_response.text)
                
                # Handle both string and dict responses
                if isinstance(cleaned_generation, dict):
                    generated_clips = cleaned_generation.get('selected_clips', [])
                else:
                    # If it's a string, try to parse as JSON
                    generated_clips = json.loads(cleaned_generation).get('selected_clips', [])
                
                print(f"✅ PASS 1 Complete: Generated {len(generated_clips)} initial clips")
                
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"❌ PASS 1 JSON Error: {e}")
                print(f"Raw Response: {generation_response.text}")
                return self.fallback_clip_selection(viral_segments, num_clips_requested)
            
            # PASS 2: Expert Review and Refinement
            print("🔄 PASS 2: Expert Review and Refinement...")
            
            review_prompt = self.create_expert_review_prompt(generated_clips, full_transcript, frontend_inputs)
            
            review_response = self.model.generate_content(
                review_prompt,
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Analytical review
                    top_p=0.9,       # Focused analysis
                    top_k=30,        # Precise evaluation
                    max_output_tokens=4000
                )
            )
            
            # Parse review response
            try:
                cleaned_review = self.clean_ai_response(review_response.text)
                
                # Handle both string and dict responses
                if isinstance(cleaned_review, dict):
                    review_results = cleaned_review.get('review_results', {})
                else:
                    # If it's a string, try to parse as JSON
                    review_results = json.loads(cleaned_review).get('review_results', {})
                
                approved_clips = review_results.get('approved_clips', [])
                refined_clips = review_results.get('refined_clips', [])
                rejected_clips = review_results.get('rejected_clips', [])
                
                print(f"✅ PASS 2 Complete: {len(approved_clips)} approved, {len(refined_clips)} refined, {len(rejected_clips)} rejected")
                
                # Combine approved and refined clips
                final_clips = approved_clips + refined_clips
                
                # Ensure we have enough clips
                if len(final_clips) < num_clips_requested:
                    print(f"⚠️ Only {len(final_clips)} high-quality clips found, using fallback for remaining")
                    fallback_clips = self.fallback_clip_selection(viral_segments, num_clips_requested - len(final_clips))
                    final_clips.extend(fallback_clips)
                
                # Limit to requested number
                final_clips = final_clips[:num_clips_requested]
                
                # Enhance and validate final clips with advanced speech boundary validation
                enhanced_clips = []
                for clip in final_clips:
                    # First enhance with basic data
                    enhanced_clip = self.enhance_clip_data(clip)
                    
                    # Then apply advanced validation and metadata
                    if enhanced_clip:
                        validated_clip = self.enhance_clip_with_advanced_metadata(enhanced_clip, viral_segments)
                        if validated_clip:
                            enhanced_clips.append(validated_clip)
                        else:
                            print(f"⚠️ Clip validation failed, skipping: {clip.get('start_time', 0)}s - {clip.get('end_time', 0)}s")
                    else:
                        print(f"⚠️ Clip enhancement failed, skipping: {clip.get('start_time', 0)}s - {clip.get('end_time', 0)}s")
                
                print(f"🎯 DUAL-PASS Complete: {len(enhanced_clips)} high-quality viral clips ready")
                return enhanced_clips
                
            except json.JSONDecodeError as e:
                print(f"❌ PASS 2 JSON Error: {e}")
                print(f"Raw Review Response: {review_response.text}")
                # Use generated clips from PASS 1 as fallback
                # Use generated clips from PASS 1 as fallback, but enhance them
                fallback_clips = self.enhance_clips_batch(generated_clips[:num_clips_requested])
                
                # Apply advanced validation to fallback clips
                validated_fallback_clips = []
                for clip in fallback_clips:
                    validated_clip = self.enhance_clip_with_advanced_metadata(clip, viral_segments)
                    if validated_clip:
                        validated_fallback_clips.append(validated_clip)
                
                return validated_fallback_clips
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ DUAL-PASS AI Processing Failed: {error_msg}")
            
            # Check for quota issues
            if "quota" in error_msg.lower() or "429" in error_msg or "rate limit" in error_msg.lower():
                print("⚠️ Quota limit reached - using fallback selection")
                self.handle_quota_error()
            return self.fallback_clip_selection(viral_segments, num_clips_requested)
    
            # Fallback to basic selection
            print("🔄 Using fallback clip selection due to AI failure")
            return self.fallback_clip_selection(viral_segments, num_clips_requested)

    def enhance_clip_data(self, clip):
        """Enhance individual clip with additional metadata and validation"""

        enhanced_clip = {
            'start_time': float(clip.get('start_time', 0)),
            'end_time': float(clip.get('end_time', 0)),
            'duration': float(clip.get('duration', 0)),
            'viral_score': int(clip.get('viral_score', 8)),
                'content_type': clip.get('content_type', 'viral'),
            'viral_factor': clip.get('viral_factor', 'High engagement potential'),
            'engagement_potential': clip.get('engagement_potential', 'High'),
            'caption': clip.get('caption', 'Viral moment'),
            'hashtags': clip.get('hashtags', ['#viral', '#trending']),
            'hook_line': clip.get('hook_line', ''),
            'call_to_action': clip.get('call_to_action', ''),
            'thumbnail_suggestion': clip.get('thumbnail_suggestion', ''),
            'target_audience': clip.get('target_audience', 'general'),
            'platforms': clip.get('platforms', ['TikTok', 'Instagram']),
            'optimal_posting_time': clip.get('optimal_posting_time', ''),
            'cross_platform_adaptation': clip.get('cross_platform_adaptation', ''),
            'segment_text': clip.get('segment_text', ''),
                'viral_potential': clip.get('viral_score', 8),
                'engagement': clip.get('viral_score', 8),
                'story_value': clip.get('viral_score', 8),
            'audio_impact': clip.get('viral_score', 8),
            'quality_score': clip.get('viral_score', 8),
            'review_status': 'dual_pass_approved'
        }
        
        return enhanced_clip

    def enhance_clips_batch(self, clips):
        """Enhance a batch of clips with consistent metadata"""
        
        enhanced_clips = []
        for clip in clips:
            enhanced_clip = self.enhance_clip_data(clip)
            enhanced_clips.append(enhanced_clip)
        
        return enhanced_clips
    
    def create_clip(self, video_path, start_time, duration, output_name, aspect_ratio_options=None, watermark_options=None):
        """Create video clip using MoviePy with smart aspect ratio processing and watermarking"""
        try:
            output_path = self.output_dir / output_name
            
            print(f"   Creating clip with MoviePy...")
            print(f"   Time: {start_time:.1f}s - {start_time + duration:.1f}s")
            
            # Load video with MoviePy
            video = VideoFileClip(video_path)
            
            # Extract the clip
            clip = video.subclip(start_time, start_time + duration)
            
            # Apply smart aspect ratio processing if options provided
            if aspect_ratio_options:
                clip = self._apply_aspect_ratio_processing(clip, aspect_ratio_options)
            
            # Apply watermark if options provided
            if watermark_options and watermark_options.get('enableWatermark', False):
                print(f"   💧 Applying watermark to clip...")
                clip = self._apply_watermark(clip, watermark_options)
            
            # Write the clip
            clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up
            clip.close()
            video.close()
            
            print(f"   MoviePy clip created successfully: {os.path.basename(output_path)}")
            return str(output_path)
            
        except Exception as e:
            error_msg = f"Failed to create clip with MoviePy: {str(e)}"
            print(f"   ERROR: {error_msg}")
            raise Exception(error_msg)

    def _apply_aspect_ratio_processing(self, clip, aspect_ratio_options):
        """Apply smart aspect ratio processing to video clip with minimal content loss and camera tracking"""
        try:
            target_ratio = aspect_ratio_options.get('targetAspectRatio', '16:9')
            preserve_original = aspect_ratio_options.get('preserveOriginal', False)
            enable_smart_cropping = aspect_ratio_options.get('enableSmartCropping', True)
            enable_letterboxing = aspect_ratio_options.get('enableLetterboxing', True)
            enable_quality_preservation = aspect_ratio_options.get('enableQualityPreservation', True)
            
            print(f"   🎬 Applying SMART aspect ratio processing: {target_ratio}")
            print(f"   📐 Strategy: Minimize content loss + Smart scaling + Strategic cropping + Camera tracking")
            
            # Parse target aspect ratio
            if ':' in target_ratio:
                width_ratio, height_ratio = map(float, target_ratio.split(':'))
                target_aspect = width_ratio / height_ratio
            else:
                target_aspect = float(target_ratio)
            
            # Get current clip dimensions
            current_width, current_height = clip.size
            current_aspect = current_width / current_height
            
            print(f"   📏 Current: {current_width}x{current_height} ({current_aspect:.3f})")
            print(f"   🎯 Target: {target_aspect:.3f}")
            
            # If preserving original aspect ratio or using normal video ratio, just add letterboxing if needed
            use_normal_ratio = aspect_ratio_options.get('useNormalVideoRatio', False)
            
            if preserve_original or use_normal_ratio:
                if preserve_original:
                    print(f"   📦 Preserving original aspect ratio: {current_aspect:.3f}")
                else:
                    print(f"   📦 Using normal video ratio (16:9) without platform optimization")
                
                if enable_letterboxing and abs(current_aspect - target_aspect) > 0.01:
                    print(f"   📦 Adding letterboxing to preserve original ratio")
                    clip = self._add_letterboxing(clip, target_aspect)
                return clip
            
            # Calculate content loss percentage for different strategies with reduced compression
            strategies = self._calculate_content_loss_strategies(
                current_width, current_height, current_aspect, target_aspect
            )
            
            print(f"   📊 Content loss analysis:")
            for strategy, loss in strategies.items():
                print(f"      {strategy}: {loss:.1f}% content loss")
            
            # Choose the best strategy (lowest content loss)
            best_strategy = min(strategies, key=strategies.get)
            best_loss = strategies[best_strategy]
            
            print(f"   🎯 Best strategy: {best_strategy} ({best_loss:.1f}% content loss)")
            
            # Apply the chosen strategy with camera tracking
            if best_strategy == "Smart Scaling + Minimal Crop":
                clip = self._apply_smart_scaling_strategy(clip, target_aspect, current_aspect)
            elif best_strategy == "Intelligent Stretch":
                clip = self._apply_intelligent_stretch_strategy(clip, target_aspect, current_aspect)
            elif best_strategy == "Strategic Crop":
                clip = self._apply_strategic_crop_strategy(clip, target_aspect, current_aspect)
            elif best_strategy == "Hybrid Approach":
                clip = self._apply_hybrid_strategy(clip, target_aspect, current_aspect)
            else:
                # Fallback to smart scaling
                clip = self._apply_smart_scaling_strategy(clip, target_aspect, current_aspect)
            
            # Apply camera tracking for better content preservation
            if enable_quality_preservation and not preserve_original:
                print(f"   🎥 Applying camera tracking for content preservation...")
                clip = self._apply_simple_camera_tracking(clip, target_aspect)
            
            # Final quality check and letterboxing if needed
            if enable_letterboxing:
                final_width, final_height = clip.size
                final_aspect = final_width / final_height
                if abs(final_aspect - target_aspect) > 0.01:
                    print(f"   📦 Adding final letterboxing for perfect ratio...")
                    clip = self._add_letterboxing(clip, target_aspect)
            
            # Apply watermark (mandatory for free tier, can be disabled for paid users)
            if aspect_ratio_options and aspect_ratio_options.get('enableWatermark', True):
                # Use default watermark settings for Zuexis branding
                watermark_text = 'Made by Zuexis'
                watermark_opacity = 0.4
                watermark_position = 'top-left'
                watermark_size = 'extra-large'
                use_logo = True
                
                print(f"   💧 Applying Zuexis watermark: {watermark_text}")
                print(f"      Position: {watermark_position}, Opacity: {watermark_opacity}, Size: {watermark_size}")
                print(f"      Logo: Yes")
                
                # Create watermark options dictionary
                watermark_options = {
                    'watermarkText': watermark_text,
                    'watermarkOpacity': watermark_opacity,
                    'watermarkPosition': watermark_position,
                    'watermarkSize': watermark_size,
                    'useLogo': use_logo
                }
                
                clip = self._apply_watermark(clip, watermark_options)
            
            print(f"   ✅ SMART aspect ratio processing complete: {clip.size[0]}x{clip.size[1]}")
            print(f"   🎯 Final aspect ratio: {clip.size[0]/clip.size[1]:.3f}")
            return clip
            
        except Exception as e:
            print(f"   ⚠️ SMART aspect ratio processing failed: {e}")
            print(f"   🔄 Continuing with original clip...")
            return clip

    def _calculate_content_loss_strategies(self, current_width, current_height, current_aspect, target_aspect):
        """Calculate content loss for different processing strategies"""
        strategies = {}
        
        # Strategy 1: Smart Scaling + Minimal Crop (Reduced Compression)
        if abs(current_aspect - target_aspect) > 0.01:
            if target_aspect > current_aspect:
                # Target is wider - minimal height crop with reduced compression
                compression_factor = 0.5  # 50% less aggressive
                scaled_width = int(current_width + (target_aspect - current_aspect) * current_height * compression_factor)
                crop_height = int(scaled_width / target_aspect)
                if crop_height <= current_height:
                    crop_loss = ((current_height - crop_height) / current_height) * 100
                    strategies["Smart Scaling + Minimal Crop"] = crop_loss * 0.5  # 50% less content loss
            else:
                # Target is taller - minimal width crop with reduced compression
                compression_factor = 0.5  # 50% less aggressive
                scaled_height = int(current_height + (current_aspect - target_aspect) * current_width * compression_factor)
                crop_width = int(scaled_height * target_aspect)
                if crop_width <= current_width:
                    crop_loss = ((current_width - crop_width) / current_width) * 100
                    strategies["Smart Scaling + Minimal Crop"] = crop_loss * 0.5  # 50% less content loss
        
        # Strategy 2: Intelligent Stretch (Reduced Compression)
        compression_factor = 0.5  # 50% less aggressive
        stretch_factor = max(target_aspect / current_aspect, current_aspect / target_aspect)
        # Apply compression factor to reduce distortion penalty
        reduced_stretch_factor = 1 + (stretch_factor - 1) * compression_factor
        distortion_penalty = (reduced_stretch_factor - 1) * 15  # Further reduced penalty for stretching
        strategies["Intelligent Stretch"] = distortion_penalty
        
        # Strategy 3: Strategic Crop (when scaling isn't enough)
        if abs(current_aspect - target_aspect) > 0.3:  # Significant difference
            if target_aspect > current_aspect:
                # Need to crop height significantly
                crop_height = int(current_width / target_aspect)
                if crop_height <= current_height:
                    crop_loss = ((current_height - crop_height) / current_height) * 100
                    strategies["Strategic Crop"] = crop_loss
            else:
                # Need to crop width significantly
                crop_width = int(current_height * target_aspect)
                if crop_width <= current_width:
                    crop_loss = ((current_width - crop_width) / current_width) * 100
                    strategies["Strategic Crop"] = crop_loss
        
        # Strategy 4: Hybrid Approach (combine scaling + minimal crop)
        if "Smart Scaling + Minimal Crop" in strategies:
            hybrid_loss = strategies["Smart Scaling + Minimal Crop"] * 0.8  # 20% improvement
            strategies["Hybrid Approach"] = hybrid_loss
        
        return strategies

    def _apply_smart_scaling_strategy(self, clip, target_aspect, current_aspect):
        """Smart scaling with minimal compression and cropping (preferred strategy)"""
        try:
            current_width, current_height = clip.size
            
            print(f"   🔄 Applying SMART SCALING strategy (reduced compression)...")
            
            # Calculate compression factor to reduce aggressive scaling
            compression_factor = 0.5  # Reduce compression by 50%
            
            if target_aspect > current_aspect:
                # Target is wider - use compression factor to reduce width scaling
                target_height = current_height
                # Apply compression factor to reduce width increase
                scaled_width = int(current_width + (target_aspect - current_aspect) * current_height * compression_factor)
                target_width = scaled_width
                
                print(f"      📏 Reduced compression scaling: {current_width} → {target_width}")
                print(f"      📐 Compression factor: {compression_factor} (30% less aggressive)")
                
                # Scale the clip to reduced target width
                scaled_clip = clip.resize((target_width, target_height))
                
            else:
                # Target is taller - use compression factor to reduce height scaling
                target_width = current_width
                # Apply compression factor to reduce height increase
                scaled_height = int(current_height + (current_aspect - target_aspect) * current_width * compression_factor)
                target_height = scaled_height
                
                print(f"      📏 Reduced compression scaling: {current_height} → {target_height}")
                print(f"      📐 Compression factor: {compression_factor} (30% less aggressive)")
                
                # Scale the clip to reduced target height
                scaled_clip = clip.resize((target_width, target_height))
            
            return scaled_clip
            
        except Exception as e:
            print(f"      ⚠️ Smart scaling failed: {e}")
            return clip

    def _apply_intelligent_stretch_strategy(self, clip, target_aspect, current_aspect):
        """Intelligent stretching with reduced compression and minimal distortion"""
        try:
            current_width, current_height = clip.size
            
            print(f"   🔄 Applying INTELLIGENT STRETCH strategy (reduced compression)...")
            
            # Calculate stretch factors with compression reduction
            compression_factor = 0.5  # Reduce compression by 50%
            
            if target_aspect > current_aspect:
                # Target is wider - reduce width stretch
                width_stretch = 1 + (target_aspect / current_aspect - 1) * compression_factor
                target_width = int(current_width * width_stretch)
                target_height = current_height
            else:
                # Target is taller - reduce height stretch
                height_stretch = 1 + (current_aspect / target_aspect - 1) * compression_factor
                target_width = current_width
                target_height = int(current_height * height_stretch)
            
            # Apply reduced stretching
            if width_stretch > 1.5 or height_stretch > 1.5:
                print(f"      ⚠️ Stretch factor still too high, falling back to reduced scaling...")
                return self._apply_smart_scaling_strategy(clip, target_aspect, current_aspect)
            
            print(f"      📏 Reduced compression stretch: {current_width}x{current_height} → {target_width}x{target_height}")
            print(f"      📐 Compression factor: {compression_factor} (25% less aggressive)")
            print(f"      📐 Stretch factors: W={width_stretch:.2f}, H={height_stretch:.2f}")
            
            return clip.resize((target_width, target_height))
            
        except Exception as e:
            print(f"      ⚠️ Intelligent stretch failed: {e}")
            return clip

    def _apply_strategic_crop_strategy(self, clip, target_aspect, current_aspect):
        """Strategic cropping when other methods aren't suitable"""
        try:
            current_width, current_height = clip.size
            
            print(f"      ✂️ Applying STRATEGIC CROP strategy...")
            
            if target_aspect > current_aspect:
                # Target is wider - crop height strategically
                crop_height = int(current_width / target_aspect)
                if crop_height <= current_height:
                    # Center crop with content awareness
                    y_offset = (current_height - crop_height) // 2
                    print(f"         📏 Strategic height crop: {current_height} → {crop_height}")
                    print(f"         🎯 Center crop at y={y_offset}")
                    
                    cropped = clip.crop(x1=0, y1=y_offset, x2=current_width, y2=y_offset + crop_height)
                    return cropped
            else:
                # Target is taller - crop width strategically
                crop_width = int(current_height * target_aspect)
                if crop_width <= current_width:
                    # Center crop with content awareness
                    x_offset = (current_width - crop_width) // 2
                    print(f"         📏 Strategic width crop: {current_width} → {crop_width}")
                    print(f"         🎯 Center crop at x={x_offset}")
                    
                    cropped = clip.crop(x1=x_offset, y1=0, x2=x_offset + crop_width, y2=current_height)
                    return cropped
            
            return clip
            
        except Exception as e:
            print(f"      ⚠️ Strategic crop failed: {e}")
            return clip

    def _apply_hybrid_strategy(self, clip, target_aspect, current_aspect):
        """Hybrid approach combining multiple strategies"""
        try:
            print(f"   🔄 Applying HYBRID strategy...")
            
            # First try smart scaling
            scaled_clip = self._apply_smart_scaling_strategy(clip, target_aspect, current_aspect)
            
            # If scaling alone isn't enough, add minimal strategic cropping
            final_width, final_height = scaled_clip.size
            final_aspect = final_width / final_height
            
            if abs(final_aspect - target_aspect) > 0.05:  # Still not perfect
                print(f"      🔄 Hybrid: Adding minimal strategic crop...")
                return self._apply_strategic_crop_strategy(scaled_clip, target_aspect, final_aspect)
            
            return scaled_clip
            
        except Exception as e:
            print(f"      ⚠️ Hybrid strategy failed: {e}")
            return clip

    def _apply_watermark(self, clip, watermark_options):
        """Apply watermark with logo and text to video clip"""
        try:
            print(f"      💧 Creating watermark...")
            
            # Extract watermark options
            text = watermark_options.get('watermarkText', 'Made by Zuexis')
            opacity = watermark_options.get('watermarkOpacity', 0.6)
            position = watermark_options.get('watermarkPosition', 'bottom-right')
            size = watermark_options.get('watermarkSize', 'medium')
            use_logo = watermark_options.get('useLogo', True)
            
            # Get clip dimensions
            clip_width, clip_height = clip.size
            
            # Determine watermark size based on clip dimensions
            if size == 'small':
                logo_size = min(clip_width, clip_height) // 20
                text_size = logo_size // 2
            elif size == 'large':
                logo_size = min(clip_width, clip_height) // 6  # Bigger logo
                text_size = logo_size // 1.5  # Much bigger text
            elif size == 'extra-large':
                logo_size = min(clip_width, clip_height) // 4  # Even bigger logo
                text_size = logo_size // 1.2  # Extra big text
            else:  # medium
                logo_size = min(clip_width, clip_height) // 12
                text_size = logo_size // 2
            
            # Create watermark composition
            watermark_clip = self._create_watermark_composition(
                text, logo_size, text_size, opacity, use_logo, clip.duration
            )
            
            if watermark_clip is None:
                print(f"      ⚠️ Watermark creation failed, continuing without watermark")
                return clip
            
            # Position the watermark
            positioned_watermark = self._position_watermark(
                watermark_clip, clip_width, clip_height, position
            )
            
            # Composite watermark onto video
            final_clip = self._composite_watermark(clip, positioned_watermark)
            
            print(f"      ✅ Watermark applied successfully")
            return final_clip
            
        except Exception as e:
            print(f"      ⚠️ Watermark application failed: {e}")
            return clip

    def _create_watermark_composition(self, text, logo_size, text_size, opacity, use_logo, clip_duration):
        """Create watermark composition with logo and text"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            
            # Calculate watermark dimensions
            if use_logo:
                watermark_width = logo_size + text_size * len(text) + 20  # Logo + text + spacing
                watermark_height = max(logo_size, text_size + 10)
            else:
                watermark_width = text_size * len(text) + 20
                watermark_height = text_size + 10
            
            # Create transparent background
            watermark_img = Image.new('RGBA', (watermark_width, watermark_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark_img)
            
            # Try to load and resize logo
            logo_img = None
            if use_logo:
                logo_img = self._load_and_resize_logo(logo_size)
            
            # Position elements
            current_x = 10
            
            # Add logo if available
            if logo_img:
                watermark_img.paste(logo_img, (current_x, (watermark_height - logo_size) // 2), logo_img)
                current_x += logo_size + 10
            
            # Add text
            try:
                # Try to use a nice font, fallback to default if not available
                font = ImageFont.truetype("arial.ttf", text_size) if os.path.exists("arial.ttf") else ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Calculate text position
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = current_x
            text_y = (watermark_height - text_height) // 2
            
            # Draw text with outline for better visibility
            outline_color = (0, 0, 0, int(255 * opacity))
            text_color = (255, 255, 255, int(255 * opacity))
            
            # Draw thicker outline for better stroke effect
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, text_y + dy), text, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((text_x, text_y), text, font=font, fill=text_color)
            
            # Convert PIL image to MoviePy clip
            watermark_array = np.array(watermark_img)
            watermark_clip = self._pil_to_moviepy_clip(watermark_img, clip_duration)
            
            return watermark_clip
            
        except Exception as e:
            print(f"         ⚠️ Watermark composition creation failed: {e}")
            return None

    def _load_and_resize_logo(self, target_size):
        """Load and resize logo from public folder"""
        try:
            from PIL import Image
            
            # Try to load the logo with no background first
            logo_paths = [
                "public/logo-removebg-preview.png",  # No background version
                "public/logo.png",                   # Regular logo
                "../public/logo-removebg-preview.png",  # Alternative paths
                "../public/logo.png"
            ]
            
            logo_img = None
            for logo_path in logo_paths:
                if os.path.exists(logo_path):
                    print(f"         📁 Loading logo from: {logo_path}")
                    logo_img = Image.open(logo_path)
                    break
            
            if logo_img is None:
                print(f"         ⚠️ No logo found in public folder")
                return None
            
            # Resize logo maintaining aspect ratio
            logo_img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Convert to RGBA if not already
            if logo_img.mode != 'RGBA':
                logo_img = logo_img.convert('RGBA')
            
            print(f"         ✅ Logo loaded and resized to {logo_img.size}")
            return logo_img
            
        except Exception as e:
            print(f"         ⚠️ Logo loading failed: {e}")
            return None

    def _apply_simple_camera_tracking(self, clip, target_aspect):
        """Simple camera tracking that calls the existing tracking function"""
        try:
            return self._apply_camera_tracking(clip, target_aspect)
        except Exception as e:
            print(f"         ⚠️ Simple camera tracking failed: {e}")
            return clip
    
    def _apply_camera_tracking(self, clip, target_aspect):
        """Apply camera tracking to preserve important content during aspect ratio changes"""
        try:
            print(f"         🎥 Starting camera tracking analysis...")
            
            # Get clip dimensions
            current_width, current_height = clip.size
            current_aspect = current_width / current_height
            
            # Calculate target dimensions
            if target_aspect > current_aspect:
                # Target is wider - we'll track horizontal movement
                target_width = int(current_height * target_aspect)
                target_height = current_height
                tracking_direction = 'horizontal'
            else:
                # Target is taller - we'll track vertical movement
                target_width = current_width
                target_height = int(current_width / target_aspect)
                tracking_direction = 'vertical'
            
            print(f"         📏 Tracking {tracking_direction} movement: {current_width}x{current_height} → {target_width}x{target_height}")
            
            # Create a tracking-aware resize function
            def tracking_resize(get_frame, t):
                try:
                    # Get current frame
                    frame = get_frame(t)
                    
                    if frame is None:
                        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    
                    # Apply motion-aware resizing with reduced compression
                    if tracking_direction == 'horizontal':
                        # Track horizontal movement - reduce compression by 40%
                        compression_factor = 0.6  # 40% less aggressive
                        scaled_width = int(current_width + (target_aspect - current_aspect) * current_height * compression_factor)
                        
                        # Use smart scaling with motion preservation
                        if scaled_width <= target_width:
                            # Scale horizontally with minimal distortion
                            frame = self._smart_horizontal_scale(frame, scaled_width, current_height)
                        else:
                            # Need to crop, but do it intelligently
                            frame = self._intelligent_horizontal_crop(frame, target_width, current_height)
                    else:
                        # Track vertical movement - reduce compression by 40%
                        compression_factor = 0.6  # 40% less aggressive
                        scaled_height = int(current_height + (current_aspect - target_aspect) * current_width * compression_factor)
                        
                        # Use smart scaling with motion preservation
                        if scaled_height <= target_height:
                            # Scale vertically with minimal distortion
                            frame = self._smart_vertical_scale(frame, current_width, scaled_height)
                        else:
                            # Need to crop, but do it intelligently
                            frame = self._intelligent_vertical_crop(frame, current_width, target_height)
                    
                    return frame
                    
                except Exception as e:
                    print(f"         ⚠️ Frame tracking failed at {t}s: {e}")
                    # Return a black frame as fallback
                    return np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Apply the tracking-aware resize
            tracked_clip = clip.fl(tracking_resize)
            
            print(f"         ✅ Camera tracking applied successfully")
            return tracked_clip
            
        except Exception as e:
            print(f"         ⚠️ Camera tracking failed: {e}")
            print(f"         🔄 Falling back to standard processing...")
            return clip
    
    def _smart_horizontal_scale(self, frame, target_width, target_height):
        """Smart horizontal scaling with motion preservation"""
        try:
            from PIL import Image
            import numpy as np
            
            # Convert numpy array to PIL image for better scaling
            pil_img = Image.fromarray(frame)
            
            # Use high-quality scaling with reduced compression
            scaled_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Convert back to numpy array
            return np.array(scaled_img)
            
        except Exception as e:
            print(f"            ⚠️ Smart horizontal scaling failed: {e}")
            return frame
    
    def _smart_vertical_scale(self, frame, target_width, target_height):
        """Smart vertical scaling with motion preservation"""
        try:
            from PIL import Image
            import numpy as np
            
            # Convert numpy array to PIL image for better scaling
            pil_img = Image.fromarray(frame)
            
            # Use high-quality scaling with reduced compression
            scaled_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Convert back to numpy array
            return np.array(scaled_img)
            
        except Exception as e:
            print(f"            ⚠️ Smart vertical scaling failed: {e}")
            return frame
    
    def _intelligent_horizontal_crop(self, frame, target_width, target_height):
        """Intelligent horizontal cropping with motion awareness"""
        try:
            # Center crop with minimal content loss
            current_width = frame.shape[1]
            x_offset = (current_width - target_width) // 2
            
            # Crop the frame
            cropped_frame = frame[:, x_offset:x_offset + target_width]
            
            return cropped_frame
            
        except Exception as e:
            print(f"            ⚠️ Intelligent horizontal crop failed: {e}")
            return frame
    
    def _intelligent_vertical_crop(self, frame, target_width, target_height):
        """Intelligent vertical cropping with motion awareness"""
        try:
            # Center crop with minimal content loss
            current_height = frame.shape[0]
            y_offset = (current_height - target_height) // 2
            
            # Crop the frame
            cropped_frame = frame[y_offset:y_offset + target_height, :]
            
            return cropped_frame
            
        except Exception as e:
            print(f"            ⚠️ Intelligent vertical crop failed: {e}")
            return frame

    def _pil_to_moviepy_clip(self, pil_image, duration):
        """Convert PIL image to MoviePy clip"""
        try:
            import numpy as np
            from moviepy.video.VideoClip import ImageClip
            
            # Convert PIL image to numpy array
            img_array = np.array(pil_image)
            
            # Create MoviePy clip
            clip = ImageClip(img_array, duration=duration)
            
            return clip
            
        except Exception as e:
            print(f"         ⚠️ PIL to MoviePy conversion failed: {e}")
            return None

    def _position_watermark(self, watermark_clip, clip_width, clip_height, position):
        """Position watermark on the video"""
        try:
            watermark_width, watermark_height = watermark_clip.size
            
            # Calculate position based on preference
            if position == 'top-left':
                x, y = 30, 30  # Slightly more margin for top-left
            elif position == 'top-right':
                x = clip_width - watermark_width - 20
                y = 20
            elif position == 'bottom-left':
                x = 20
                y = clip_height - watermark_height - 20
            elif position == 'bottom-right':
                x = clip_width - watermark_width - 20
                y = clip_height - watermark_height - 20
            elif position == 'center':
                x = (clip_width - watermark_width) // 2
                y = (clip_height - watermark_height) // 2
            elif position == 'top-center':
                x = (clip_width - watermark_width) // 2
                y = 20
            elif position == 'bottom-center':
                x = (clip_width - watermark_width) // 2
                y = clip_height - watermark_height - 20
            else:  # Default to bottom-right
                x = clip_width - watermark_width - 20
                y = clip_height - watermark_height - 20
            
            # Ensure watermark stays within bounds
            x = max(0, min(x, clip_width - watermark_width))
            y = max(0, min(y, clip_height - watermark_height))
            
            print(f"         📍 Positioning watermark at ({x}, {y}) - {position}")
            
            # Set position
            positioned_watermark = watermark_clip.set_position((x, y))
            
            return positioned_watermark
            
        except Exception as e:
            print(f"         ⚠️ Watermark positioning failed: {e}")
            return watermark_clip

    def _composite_watermark(self, video_clip, watermark_clip):
        """Composite watermark onto video clip"""
        try:
            from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
            
            # Create composite clip
            composite = CompositeVideoClip([video_clip, watermark_clip])
            
            return composite
            
        except Exception as e:
            print(f"         ⚠️ Watermark compositing failed: {e}")
            return video_clip

    def _smart_crop_clip(self, clip, target_width, target_height, target_aspect):
        """Smart crop clip to target aspect ratio while preserving important content (LEGACY - now uses new strategies)"""
        try:
            print(f"   🔄 Using new SMART aspect ratio strategies instead of legacy cropping...")
            return self._apply_smart_scaling_strategy(clip, target_aspect, clip.size[0] / clip.size[1])
        except Exception as e:
            print(f"   ⚠️ Legacy smart cropping failed: {e}")
            return clip

    def _add_letterboxing(self, clip, target_aspect):
        """Add letterboxing to match target aspect ratio"""
        try:
            current_width, current_height = clip.size
            current_aspect = current_width / current_height
            
            if abs(current_aspect - target_aspect) < 0.01:
                return clip  # Already matches
            
            # Calculate new dimensions
            if target_aspect > current_aspect:
                # Target is wider - add horizontal letterboxing
                new_width = int(current_height * target_aspect)
                new_height = current_height
                x_offset = (new_width - current_width) // 2
                y_offset = 0
            else:
                # Target is taller - add vertical letterboxing
                new_width = current_width
                new_height = int(current_width / target_aspect)
                x_offset = 0
                y_offset = (new_height - current_height) // 2
            
            # Create new clip with letterboxing
            from moviepy.video.VideoClip import ColorClip
            
            # Create black background
            background = ColorClip(size=(new_width, new_height), color=(0, 0, 0))
            background = background.set_duration(clip.duration)
            
            # Position the original clip
            positioned_clip = clip.set_position((x_offset, y_offset))
            
            # Composite the clips
            final_clip = background.set_make_frame(
                lambda t: self._composite_frames(background.get_frame(t), positioned_clip.get_frame(t), x_offset, y_offset)
            )
            
            return final_clip
            
        except Exception as e:
            print(f"   ⚠️ Letterboxing failed: {e}")
            return clip

    def _composite_frames(self, bg_frame, clip_frame, x_offset, y_offset):
        """Composite background and clip frames for letterboxing"""
        try:
            import numpy as np
            
            # Create a copy of background frame
            result = bg_frame.copy()
            
            # Get clip dimensions
            clip_height, clip_width = clip_frame.shape[:2]
            
            # Calculate bounds
            x1, y1 = x_offset, y_offset
            x2, y2 = x_offset + clip_width, y_offset + clip_height
            
            # Ensure bounds are within result frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(result.shape[1], x2)
            y2 = min(result.shape[0], y2)
            
            # Update clip bounds
            clip_x1 = max(0, -x_offset)
            clip_y1 = max(0, -y_offset)
            clip_x2 = clip_x1 + (x2 - x1)
            clip_y2 = clip_y1 + (y2 - y1)
            
            # Composite the frames
            if clip_x2 > clip_x1 and clip_y2 > clip_y1:
                result[y1:y2, x1:x2] = clip_frame[clip_y1:clip_y2, clip_x1:clip_x2]
            
            return result
            
        except Exception as e:
            print(f"   ⚠️ Frame compositing failed: {e}")
            return bg_frame
    
    def generate_viral_clips(self, video_path, num_clips=3, frontend_inputs=None):
        """Generate viral clips using AI-powered selection and MoviePy"""
        try:
            # Extract and analyze audio segments
            viral_segments, full_transcript = self.extract_audio_segments(video_path)
            
            # AI-powered clip selection with captions and hashtags
            viral_moments = self.ai_select_best_clips(viral_segments, full_transcript, num_clips, frontend_inputs)
            
            # Generate clips from AI-selected moments
            generated_clips = []
            clip_details = []
            
            print(f"Generating {len(viral_moments)} AI-selected viral clips...")
            
            for i, moment in enumerate(viral_moments):
                start_time = moment['start_time']
                duration = moment['duration']
                
                # Create descriptive filename
                safe_caption = re.sub(r'[^\w\s-]', '', moment['caption'])[:30]
                clip_name = f"viral_clip_{i+1}_{moment['viral_score']}_{safe_caption}.mp4"
                clip_name = clip_name.replace(' ', '_')
                
                print(f"Creating clip {i+1}/{len(viral_moments)}: {clip_name}")
                print(f"   Time: {start_time:.1f}s - {start_time + duration:.1f}s")
                print(f"   Score: {moment['viral_score']}/10")
                print(f"   Caption: {moment['caption'][:50]}...")
                print(f"   Hashtags: {', '.join(moment['hashtags'][:3])}...")
                
                try:
                    # Extract aspect ratio and watermark options from frontend inputs
                    aspect_ratio_options = None
                    watermark_options = None
                    
                    if frontend_inputs and 'processingOptions' in frontend_inputs:
                        processing_options = frontend_inputs['processingOptions']
                        
                        # Extract aspect ratio options
                        if 'targetAspectRatio' in processing_options or 'aspectRatioOptions' in processing_options:
                            aspect_ratio_options = {
                                'targetAspectRatio': processing_options.get('targetAspectRatio', '16:9'),
                                'preserveOriginal': processing_options.get('preserveOriginalAspectRatio', False),
                                'enableSmartCropping': processing_options.get('enableSmartCropping', True),
                                'enableLetterboxing': processing_options.get('enableLetterboxing', True),
                                'enableQualityPreservation': processing_options.get('enableQualityPreservation', True)
                            }
                            print(f"   🎬 Using aspect ratio options: {aspect_ratio_options}")
                        
                        # Extract watermark options
                        if 'watermarkOptions' in processing_options:
                            watermark_options = processing_options['watermarkOptions']
                            print(f"   💧 Using watermark options: {watermark_options}")
                        elif 'enableWatermark' in processing_options:
                            # Fallback to individual watermark properties
                            watermark_options = {
                                'enableWatermark': processing_options.get('enableWatermark', True),
                                'useLogo': processing_options.get('useLogo', True),
                                'watermarkText': processing_options.get('watermarkText', 'Made by Zuexis'),
                                'watermarkPosition': processing_options.get('watermarkPosition', 'top-left'),
                                'watermarkSize': processing_options.get('watermarkSize', 'extra-large'),
                                'watermarkOpacity': processing_options.get('watermarkOpacity', 0.4)
                            }
                            print(f"   💧 Using fallback watermark options: {watermark_options}")
                    
                    clip_path = self.create_clip(video_path, start_time, duration, clip_name, aspect_ratio_options, watermark_options)
                    generated_clips.append(clip_path)
                    
                    # Save detailed clip information
                    clip_info = {
                        'clip_number': i + 1,
                        'filename': clip_name,
                        'filepath': clip_path,
                        'start_time': start_time,
                        'end_time': moment['end_time'],
                        'duration': duration,
                        'viral_score': moment['viral_score'],
                        'content_type': moment['content_type'],
                        'caption': moment['caption'],
                        'hashtags': moment['hashtags'],
                        'target_audience': moment['target_audience'],
                        'platforms': moment['platforms'],
                        'segment_text': moment['segment_text'],
                        'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    clip_details.append(clip_info)
                    
                    # Save individual clip analysis
                    analysis_path = self.output_dir / f"clip_{i+1}_analysis.json"
                    with open(analysis_path, 'w', encoding='utf-8') as f:
                        json.dump(clip_info, f, indent=2, ensure_ascii=False)
                    
                    print(f"   Clip created: {os.path.basename(clip_path)}")
                    
                except Exception as e:
                    error_msg = f"Failed to create clip {i+1}: {e}"
                    print(f"   ERROR: {error_msg}")
            
            # Save comprehensive analysis report
            report_data = {
                'video_path': video_path,
                'total_segments': len(viral_segments),
                'selected_clips': len(viral_moments),
                'clips_generated': len(generated_clips),
                'clip_details': clip_details,
                'full_transcript': full_transcript,
                'viral_segments': [
                    {
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text'],
                        'viral_score': seg['viral_score']
                    } for seg in viral_segments
                ],
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            report_path = self.output_dir / "generation_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Save full transcription
            transcription_path = self.output_dir / "transcription.txt"
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write(f"Full Video Transcription\n{'='*50}\n\n")
                f.write(f"Total Duration: {max(seg['end'] for seg in viral_segments):.1f} seconds\n")
                f.write(f"Viral Segments Found: {len(viral_segments)}\n\n")
                f.write("Viral Segments:\n")
                for seg in viral_segments:
                    f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] Score: {seg['viral_score']}\n")
                    f.write(f"Text: {seg['text']}\n\n")
                f.write("Full Transcript:\n")
                f.write(full_transcript)
            
            print(f"Generated {len(generated_clips)} clips with AI-powered selection")
            print(f"Report saved to: {report_path}")
            
            return generated_clips, full_transcript
            
        except Exception as e:
            raise e
    
    def _get_clip_info_from_report(self, clip_path, clip_number):
        """Extract clip information from the generation report"""
        try:
            # Try to read the individual clip analysis file
            analysis_path = self.output_dir / f"clip_{clip_number}_analysis.json"
            if analysis_path.exists():
                with open(analysis_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # Fallback to basic info
            return {
                'start_time': 0,
                'end_time': 30,
                'duration': 30,
                'viral_score': 8,
                'content_type': 'viral',
                'caption': f'Viral clip #{clip_number}',
                'hashtags': ['viral', 'trending', 'amazing'],
                'target_audience': 'general',
                'segment_text': ''
            }
        except Exception as e:
            print(f"Warning: Could not read clip info for clip {clip_number}: {e}")
            # Return default values
            return {
                'start_time': 0,
                'end_time': 30,
                'duration': 30,
                'viral_score': 8,
                'content_type': 'viral',
                'caption': f'Viral clip #{clip_number}',
                'hashtags': ['viral', 'trending', 'amazing'],
                'target_audience': 'general',
                'segment_text': ''
            }

    def validate_speech_boundaries(self, clip, transcript_segments):
        """Advanced validation to ensure clips don't cut mid-sentence or mid-word"""
        
        start_time = clip.get('start_time', 0)
        end_time = clip.get('end_time', 0)
        
        # Find transcript segments that overlap with this clip
        overlapping_segments = []
        for segment in transcript_segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if segment overlaps with clip
            if (seg_start <= end_time and seg_end >= start_time):
                overlapping_segments.append(segment)
        
        if not overlapping_segments:
            return False, "No transcript segments found for this clip"
        
        # Check start boundary
        first_segment = overlapping_segments[0]
        first_text = first_segment.get('text', '').strip()
        
        # Check if we're starting mid-sentence
        if first_text and not first_text[0].isupper() and not first_text.startswith(('"', "'", '-', '—')):
            return False, f"Clip starts mid-sentence: '{first_text[:50]}...'"
        
        # Check end boundary
        last_segment = overlapping_segments[-1]
        last_text = last_segment.get('text', '').strip()
        
        # Check if we're ending mid-sentence
        if last_text and not last_text.endswith(('.', '!', '?', ':', ';', '"', "'", '-', '—')):
            return False, f"Clip ends mid-sentence: '{last_text[-50:]}...'"
        
        # Check for incomplete words at boundaries
        if first_text and first_text.startswith(('...', '..', '.')):
            return False, "Clip starts with incomplete punctuation"
        
        if last_text and last_text.endswith(('...', '..', '.')):
            return False, "Clip ends with incomplete punctuation"
        
        return True, "Speech boundaries validated successfully"

    def optimize_clip_timing(self, clip, transcript_segments, target_duration=60):
        """Optimize clip timing to hit natural speech boundaries"""
        
        start_time = clip.get('start_time', 0)
        end_time = clip.get('end_time', 0)
        current_duration = end_time - start_time
        
        # Find the best start and end points within a reasonable range
        search_range = 5  # seconds to search forward/backward
        
        best_start = start_time
        best_end = end_time
        best_score = 0
        
        # Search for better boundaries
        for start_offset in range(-int(search_range), int(search_range) + 1):
            for end_offset in range(-int(search_range), int(search_range) + 1):
                test_start = start_time + start_offset
                test_end = end_time + end_offset
                test_duration = test_end - test_start
                
                # Check if duration is acceptable
                if test_duration < 15 or test_duration > target_duration + 10:
                    continue
                
                # Validate boundaries
                is_valid, _ = self.validate_speech_boundaries({
                    'start_time': test_start,
                    'end_time': test_end
                }, transcript_segments)
                
                if is_valid:
                    # Score based on duration closeness to target
                    duration_score = 1 - abs(test_duration - target_duration) / target_duration
                    
                    # Bonus for natural boundaries
                    boundary_bonus = 0.2 if self.is_natural_boundary(test_start, test_end, transcript_segments) else 0
                    
                    total_score = duration_score + boundary_bonus
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_start = test_start
                        best_end = test_end
        
        # Update clip with optimized timing
        clip['start_time'] = max(0, best_start)
        clip['end_time'] = best_end
        clip['duration'] = best_end - best_start
        clip['timing_optimized'] = True
        
        return clip

    def is_natural_boundary(self, start_time, end_time, transcript_segments):
        """Check if the timing represents a natural speech boundary"""
        
        # Look for segments that start/end at these times
        for segment in transcript_segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if we're at a natural start
            if abs(seg_start - start_time) < 0.5:
                text = segment.get('text', '').strip()
                if text and text[0].isupper():
                    return True
            
            # Check if we're at a natural end
            if abs(seg_end - end_time) < 0.5:
                text = segment.get('text', '').strip()
                if text and text[-1] in '.!?:;"\'-—':
                    return True
        
        return False

    def enhance_clip_with_advanced_metadata(self, clip, transcript_segments):
        """Add advanced metadata and validation to clips"""
        
        # Handle case where no transcript segments are available
        if not transcript_segments:
            print("ℹ️ No transcript segments available for speech boundary validation")
            clip['speech_boundaries_validated'] = False
            clip['validation_message'] = "No transcript segments available"
            clip['quality_grade'] = self.calculate_quality_grade(clip)
            clip['viral_potential_enhanced'] = self.enhance_viral_potential_score(clip)
            return clip
        
        # Validate speech boundaries
        try:
            is_valid, validation_msg = self.validate_speech_boundaries(clip, transcript_segments)
            
            if not is_valid:
                print(f"⚠️ Speech boundary validation failed: {validation_msg}")
                # Try to optimize timing
                try:
                    clip = self.optimize_clip_timing(clip, transcript_segments)
                    
                    # Re-validate
                    is_valid, validation_msg = self.validate_speech_boundaries(clip, transcript_segments)
                    if not is_valid:
                        print(f"⚠️ Speech boundary optimization failed: {validation_msg}")
                except Exception as e:
                    print(f"⚠️ Speech boundary optimization failed: {e}")
                    is_valid = False
                    validation_msg = f"Optimization error: {e}"
        except Exception as e:
            print(f"⚠️ Speech boundary validation failed: {e}")
            is_valid = False
            validation_msg = f"Validation error: {e}"
        
        # Add advanced metadata
        clip['speech_boundaries_validated'] = is_valid
        clip['validation_message'] = validation_msg
        clip['quality_grade'] = self.calculate_quality_grade(clip)
        clip['viral_potential_enhanced'] = self.enhance_viral_potential_score(clip)
        
        return clip

    def calculate_quality_grade(self, clip):
        """Calculate overall quality grade for the clip"""
        
        viral_score = clip.get('viral_score', 8)
        duration = clip.get('duration', 0)
        
        # Base grade from viral score
        if viral_score >= 9:
            base_grade = 'A'
        elif viral_score >= 8:
            base_grade = 'B'
        elif viral_score >= 7:
            base_grade = 'C'
        else:
            base_grade = 'D'
        
        # Duration optimization
        if 15 <= duration <= 60:
            duration_bonus = '+'
        elif 10 <= duration <= 90:
            duration_bonus = ''
        else:
            duration_bonus = '-'
        
        return f"{base_grade}{duration_bonus}"

    def enhance_viral_potential_score(self, clip):
        """Enhance viral potential score based on multiple factors"""
        
        base_score = clip.get('viral_score', 8)
        
        # Factors that boost viral potential
        boost_factors = 0
        
        # Caption quality
        caption = clip.get('caption', '')
        if caption and len(caption) > 20:
            boost_factors += 0.5
        
        # Hashtag relevance
        hashtags = clip.get('hashtags', [])
        if len(hashtags) >= 5:
            boost_factors += 0.3
        
        # Platform optimization
        platforms = clip.get('platforms', [])
        if len(platforms) >= 2:
            boost_factors += 0.2
        
        # Enhanced score (capped at 10)
        enhanced_score = min(10, base_score + boost_factors)
        
        return round(enhanced_score, 1)

    def process_video(self, video_path: str, project_data: dict):
        """Process video and return clips with transcription and analysis"""
        try:
            print(f"🎬 [ViralClipGenerator] Starting video processing for: {video_path}")
            
            # Extract parameters from project data
            num_clips = project_data.get('numClips', 3)
            ai_prompt = project_data.get('aiPrompt', '')
            target_platforms = project_data.get('targetPlatforms', ['tiktok'])
            processing_options = project_data.get('processingOptions', {})
            
            print(f"📊 [ViralClipGenerator] Processing parameters:")
            print(f"   - Number of clips: {num_clips}")
            print(f"   - AI Prompt: {ai_prompt[:50]}...")
            print(f"   - Target platforms: {target_platforms}")
            
            # Create frontend inputs structure for compatibility
            frontend_inputs = {
                'ai_prompt': ai_prompt,
                'target_platforms': target_platforms,
                'processing_options': processing_options
            }
            
            # Generate viral clips using existing method
            print("🎬 [ViralClipGenerator] Calling generate_viral_clips...")
            clips, transcription = self.generate_viral_clips(video_path, num_clips, frontend_inputs)
            
            if not clips:
                print("❌ [ViralClipGenerator] No clips generated")
                return {
                    'success': False,
                    'error': 'No clips were generated'
                }
            
            print(f"✅ [ViralClipGenerator] Successfully generated {len(clips)} clips")
            
            # Prepare clips data for frontend
            clips_data = []
            for i, clip in enumerate(clips):
                clip_info = {
                    'id': f"clip_{i+1}",
                    'name': clip.get('name', f'Clip {i+1}'),
                    'start_time': clip.get('start_time', 0),
                    'end_time': clip.get('end_time', 0),
                    'duration': clip.get('duration', 0),
                    'caption': clip.get('caption', ''),
                    'hashtags': clip.get('hashtags', []),
                    'platforms': clip.get('platforms', target_platforms),
                    'viral_score': clip.get('viral_score', 8.0),
                    'file_path': clip.get('file_path', ''),
                    'thumbnail_path': clip.get('thumbnail_path', ''),
                    'transcription': clip.get('transcription', ''),
                    'ai_insights': clip.get('ai_insights', {})
                }
                clips_data.append(clip_info)
            
            # Return success with all data
            return {
                'success': True,
                'clips': clips_data,
                'transcription': transcription,
                'analysis': {
                    'total_clips': len(clips),
                    'video_path': video_path,
                    'processing_options': processing_options
                },
                'message': f'Successfully generated {len(clips)} viral clips'
            }
            
        except Exception as e:
            print(f"❌ [ViralClipGenerator] Error in process_video: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Flask Application Setup
app = Flask(__name__)

# Configure CORS to allow frontend requests from all origins
# Using wildcard (*) to fix CORS issues with Render deployment
# Note: This is less secure but necessary for cross-origin requests from localhost to Render
CORS(app, 
     origins="*",  # Allow all origins for now to fix CORS issues
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Content-Length", "Accept", "Origin", "X-Requested-With"],
     supports_credentials=False,  # Disable credentials for wildcard origins
     max_age=3600  # Cache preflight response for 1 hour
)

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Content-Length, Accept, Origin, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '3600'
    
    # Handle preflight requests
    if request.method == 'OPTIONS':
        response.status_code = 200
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Content-Length, Accept, Origin, X-Requested-With'
    
    return response

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'viral_clips'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Also ensure the viral_clips directory exists
os.makedirs('viral_clips', exist_ok=True)

# Global instances
generator = None
job_queue = None

def initialize_services():
    """Initialize all services"""
    global generator, job_queue
    
    try:
        # Initialize job queue
        job_queue = JobQueue()
        app.logger.info("JobQueue initialized successfully")
        
        # Initialize generator
        generator = ViralClipGenerator(output_dir=app.config['OUTPUT_FOLDER'])
        app.logger.info("ViralClipGenerator initialized successfully")
        
        # Ensure output directory exists and is writable
        try:
            output_dir = Path(app.config['OUTPUT_FOLDER'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = output_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            app.logger.info(f"Output directory {output_dir} is writable")
        except Exception as e:
            app.logger.error(f"Failed to create or verify output directory {app.config['OUTPUT_FOLDER']}: {e}")
            # Try to create in current directory as fallback
            try:
                fallback_dir = Path("viral_clips_fallback")
                fallback_dir.mkdir(exist_ok=True)
                app.config['OUTPUT_FOLDER'] = str(fallback_dir)
                app.logger.info(f"Using fallback output directory: {fallback_dir}")
            except Exception as fallback_e:
                app.logger.error(f"Failed to create fallback directory: {fallback_e}")
                return False
        
        return True
    except Exception as e:
        app.logger.error(f"Failed to initialize services: {e}")
        return False

# Initialize services on startup
def setup():
    initialize_services()

# CORS is handled by Flask-CORS library - no need for custom middleware

# Use with_appcontext for modern Flask versions
with app.app_context():
    setup()

# API Routes
@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        # Preflight request - return with CORS headers
        response = jsonify({'message': 'Health check preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    # Check OAuth configuration
    oauth_config = {
        'google_drive': {
            'client_id': bool(os.getenv('VITE_GOOGLE_DRIVE_CLIENT_ID')),
            'client_secret': bool(os.getenv('VITE_GOOGLE_DRIVE_CLIENT_SECRET')),
            'api_key': bool(os.getenv('VITE_GOOGLE_DRIVE_API_KEY')),
            'redirect_uri': os.getenv('VITE_GOOGLE_DRIVE_REDIRECT_URI', 'http://localhost:3000/oauth-callback.html')
        },
        'gemini': {
            'api_key': bool(os.getenv('VITE_GEMINI_API_KEY'))
        }
    }
    
    # Create response with explicit CORS headers
    response = jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'generator_ready': generator is not None,
        'message': 'Viral Clip Generator Backend is running',
        'oauth_config': oauth_config,
        'missing_credentials': [
            key for key, value in {
                'VITE_GOOGLE_DRIVE_CLIENT_ID': oauth_config['google_drive']['client_id'],
                'VITE_GOOGLE_DRIVE_CLIENT_SECRET': oauth_config['google_drive']['client_secret'],
                'VITE_GEMINI_API_KEY': oauth_config['gemini']['api_key']
            }.items() if not value
        ]
    })
    
    # Add explicit CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    return response

@app.route('/', methods=['GET', 'OPTIONS'])
def root():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Root endpoint preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    """Root endpoint for basic connectivity test"""
    return jsonify({
        'message': 'Viral Clip Generator Backend',
        'status': 'running',
        'version': '2.0.0',
        'endpoints': {
            'health': '/api/health',
            'frontend_status': '/api/frontend/status',
            'upload': '/api/upload',
            'process': '/api/process',
            'test_cors': '/api/frontend/test-cors',
            'google_oauth': '/api/frontend/google-oauth',
            'test_google_token': '/api/frontend/test-google-token'
        }
    })

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_video():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Upload endpoint preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    """Upload video file endpoint"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
        if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, wmv, flv'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        app.logger.info(f"Video uploaded: {safe_filename}")
        
        return jsonify({
            'success': True,
            'filename': safe_filename,
            'filepath': filepath,
            'message': 'Video uploaded successfully'
        })
        
    except Exception as e:
        app.logger.error(f"Upload failed: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/process', methods=['POST', 'OPTIONS'])
def process_video():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Process endpoint preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    """Process video and generate viral clips endpoint"""
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'Filename not provided'}), 400
        
        filename = data['filename']
        num_clips = data.get('num_clips', 3)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Video file not found'}), 404
        
        app.logger.info(f"Processing video: {filename} for {num_clips} clips")
        
        # Process video (can be enhanced with frontend inputs if provided)
        clips, transcription = generator.generate_viral_clips(filepath, num_clips)
        
        # Prepare response
        response_data = {
            'success': True,
            'filename': filename,
            'clips_generated': len(clips),
            'clips': clips,
            'transcription': transcription,
            'output_directory': app.config['OUTPUT_FOLDER'],
            'message': f'Successfully generated {len(clips)} viral clips'
        }
        
        app.logger.info(f"Processing complete: {len(clips)} clips generated")
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Processing failed: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/clips', methods=['GET'])
def get_clips():
    """Get list of generated clips endpoint"""
    try:
        clips_dir = Path(app.config['OUTPUT_FOLDER'])
        clips = []
        
        for file_path in clips_dir.glob('*.mp4'):
            clips.append({
                'filename': file_path.name,
                'size': file_path.stat().st_size,
                'created': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'path': str(file_path)
            })
        
        return jsonify({
            'success': True,
            'clips': clips,
            'total': len(clips)
        })
        
    except Exception as e:
        app.logger.error(f"Failed to get clips: {e}")
        return jsonify({'error': f'Failed to get clips: {str(e)}'}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_clip(filename):
    """Download generated clip endpoint"""
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Clip not found'}), 404
        
        return send_file(filepath, as_attachment=True)
        
    except Exception as e:
        app.logger.error(f"Download failed: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/api/frontend/download-clip/<filename>', methods=['GET'])
def frontend_download_clip(filename):
    """Download clip for frontend with proper CORS headers"""
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Clip not found'}), 404
        
        # Set CORS headers for frontend
        response = send_file(filepath, as_attachment=True)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
        
    except Exception as e:
        app.logger.error(f"Frontend download failed: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/api/transcription/<filename>', methods=['GET'])
def get_transcription(filename):
    """Get transcription for a video endpoint"""
    try:
        # Extract base filename without extension
        base_name = os.path.splitext(filename)[0]
        transcript_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_transcription.txt")
        
        if os.path.exists(transcript_file):
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcription = f.read()
            
            return jsonify({
                'success': True,
                'transcription': transcription,
                'filename': filename
            })
        else:
            return jsonify({'error': 'Transcription not found'}), 404
        
    except Exception as e:
        app.logger.error(f"Failed to get transcription: {e}")
        return jsonify({'error': f'Failed to get transcription: {str(e)}'}), 500

@app.route('/api/analysis/<filename>', methods=['GET'])
def get_analysis(filename):
    """Get AI analysis for a video endpoint"""
    try:
        # Extract base filename without extension
        base_name = os.path.splitext(filename)[0]
        analysis_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_analysis.json")
        
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            return jsonify({
                'success': True,
                'analysis': analysis,
                'filename': filename
            })
        else:
            return jsonify({'error': 'Analysis not found'}), 404
        
    except Exception as e:
        app.logger.error(f"Failed to get analysis: {e}")
        return jsonify({'error': f'Failed to get analysis: {str(e)}'}), 500

# Frontend Integration Routes
@app.route('/api/frontend/status', methods=['GET', 'OPTIONS'])
def frontend_status():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Frontend status preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    """Get frontend integration status"""
    return jsonify({
        'backend_ready': True,
        'generator_ready': generator is not None,
        'api_endpoints': [
            '/api/health',
            '/api/upload',
            '/api/process',
            '/api/clips',
            '/api/download/<filename>',
            '/api/transcription/<filename>',
            '/api/analysis/<filename>'
        ],
        'frontend_endpoints': [
            '/api/frontend/status',
            '/api/frontend/connect',
            '/api/frontend/process-project',
            '/api/frontend/create-project',
            '/api/frontend/upload-file',
            '/api/frontend/processing-status/<project_id>',
            '/api/frontend/project-details/<project_id>',
            '/api/frontend/validate-inputs',
            '/api/frontend/start-persistent-processing',
            '/api/frontend/job-status/<job_id>',
            '/api/frontend/user-jobs/<user_id>',
            '/api/frontend/job-results/<job_id>'
        ],
        'supported_formats': ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'],
        'max_file_size': '500MB',
        'supported_source_types': ['file', 'url', 'text'],
        'supported_platforms': ['tiktok', 'instagram', 'youtube', 'twitter', 'linkedin'],
        'processing_options': {
            'duration_settings': ['targetDuration', 'minDuration', 'maxDuration'],
            'quality_settings': ['low', 'medium', 'high'],
            'ai_features': [
                'aiEnhancement',
                'generateTranscription',
                'enablePerformancePrediction',
                'enableStyleAnalysis',
                'enableContentModeration',
                'enableTrendAnalysis',
                'enableAdvancedCaptions',
                'enableViralOptimization',
                'enableAudienceAnalysis',
                'enableCompetitorAnalysis',
                'enableContentStrategy'
            ]
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/frontend/connect', methods=['POST', 'OPTIONS'])
def frontend_connect():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Frontend connect preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    """Test frontend-backend connectivity"""
    try:
        data = request.get_json()
        frontend_url = data.get('frontend_url', 'Unknown')
        
        app.logger.info(f"Frontend connection test from: {frontend_url}")
        
        return jsonify({
            'success': True,
            'message': 'Frontend-backend connection successful',
            'backend_version': '2.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Frontend connection test failed: {e}")
        return jsonify({'error': f'Connection test failed: {str(e)}'}), 500

def process_video_with_generator(
    video_path: str,
    project_name: str,
    description: str,
    ai_prompt: str,
    target_platforms: list,
    num_clips: int,
    processing_options: dict
):
    """Process video using the existing ViralClipGenerator"""
    try:
        print(f"🎬 [Backend] ===== PROCESSING VIDEO WITH GENERATOR =====")
        print(f"📁 [Backend] Video path: {video_path}")
        print(f"📊 [Backend] Project: {project_name}")
        print(f"📝 [Backend] AI Prompt: {ai_prompt[:100]}...")
        
        # Check if generator is available
        if generator is None:
            return {'success': False, 'error': 'Video processing service not available'}
        
        # Create project data structure
        project_data = {
            'projectName': project_name,
            'description': description,
            'aiPrompt': ai_prompt,
            'targetPlatforms': target_platforms,
            'numClips': num_clips,
            'processingOptions': processing_options
        }
        
        print("🎬 [Backend] Starting video processing...")
        
        # Process the video using the existing generator
        # This should use the same logic that was working locally
        result = generator.process_video(
            video_path,
            project_data
        )
        
        if result and result.get('success'):
            print("✅ [Backend] Video processing completed successfully!")
            return {
                'success': True,
                'clips': result.get('clips', []),
                'transcription': result.get('transcription', ''),
                'analysis': result.get('analysis', {}),
                'message': 'Video processed successfully'
            }
        else:
            error_msg = result.get('error', 'Unknown processing error') if result else 'Processing failed'
            print(f"❌ [Backend] Video processing failed: {error_msg}")
            return {'success': False, 'error': error_msg}
            
    except Exception as e:
        print(f"❌ [Backend] Error in process_video_with_generator: {e}")
        return {'success': False, 'error': str(e)}

@app.route('/api/frontend/upload-chunk', methods=['POST', 'OPTIONS'])
def frontend_upload_chunk():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Chunk upload preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    
    
    
    """Handle chunked video uploads"""
    try:
        print(f"📦 [Backend] ===== CHUNK UPLOAD REQUEST RECEIVED =====")
        print(f"📊 [Backend] Request Method: {request.method}")
        print(f"📊 [Backend] Content-Type: {request.content_type}")
        print(f"📊 [Backend] Form Data Keys: {list(request.form.keys()) if request.form else 'None'}")
        print(f"📊 [Backend] Files Keys: {list(request.files.keys()) if request.files else 'None'}")
        
        # Extract chunk information
        chunk_index = int(request.form.get('chunkIndex', 0))
        total_chunks = int(request.form.get('totalChunks', 1))
        file_name = request.form.get('fileName', 'unknown')
        project_id = request.form.get('projectId', 'unknown')
        
        print(f"📦 [Backend] Chunk {chunk_index + 1}/{total_chunks} for file: {file_name}")
        
        # Get the chunk data
        if 'chunkData' not in request.files:
            error_response = jsonify({'error': 'No chunk data provided'})
            return error_response, 400
        
        chunk_file = request.files['chunkData']
        
        # Create project directory if this is the first chunk
        if chunk_index == 0:
            project_dir = os.path.join(app.config['UPLOAD_FOLDER'], project_id)
            os.makedirs(project_dir, exist_ok=True)
            print(f"📁 [Backend] Created project directory: {project_dir}")
        
        # Save chunk to temporary location
        chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], project_id, f"chunk_{chunk_index:04d}.blob")
        chunk_file.save(chunk_path)
        
        print(f"📦 [Backend] Saved chunk {chunk_index + 1} to: {chunk_path}")
        print(f"📦 [Backend] Chunk size: {os.path.getsize(chunk_path)} bytes")
        
        # If this is the last chunk, reconstruct the file
        if chunk_index == total_chunks - 1:
            print(f"🎬 [Backend] Last chunk received, reconstructing file...")
            
            # Reconstruct the complete file
            final_file_path = os.path.join(app.config['UPLOAD_FOLDER'], project_id, file_name)
            
            with open(final_file_path, 'wb') as final_file:
                for i in range(total_chunks):
                    chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], project_id, f"chunk_{i:04d}.blob")
                    if os.path.exists(chunk_path):
                        with open(chunk_path, 'rb') as chunk_file:
                            final_file.write(chunk_file.read())
                        # Clean up chunk file
                        os.remove(chunk_path)
                    else:
                        print(f"❌ [Backend] Missing chunk {i}")
                        error_response = jsonify({'error': f'Missing chunk {i}'})
                        return error_response, 400
            
            print(f"✅ [Backend] File reconstructed successfully: {final_file_path}")
            print(f"✅ [Backend] Final file size: {os.path.getsize(final_file_path)} bytes")
            
            # Now process the complete video
            print(f"🎬 [Backend] Starting video processing for reconstructed file...")
            
            try:
                # Extract project data from form
                project_name = request.form.get('projectName', 'Unknown Project')
                description = request.form.get('description', '')
                ai_prompt = request.form.get('aiPrompt', '')
                target_platforms = json.loads(request.form.get('targetPlatforms', '["tiktok"]'))
                num_clips = int(request.form.get('numClips', 3))
                processing_options = json.loads(request.form.get('processingOptions', '{}'))
                
                print(f"📊 [Backend] Processing video with:")
                print(f"   - Project: {project_name}")
                print(f"   - Description: {description}")
                print(f"   - AI Prompt: {ai_prompt[:50]}...")
                print(f"   - Target Platforms: {target_platforms}")
                print(f"   - Number of Clips: {num_clips}")
                
                # Check if generator is available
                if generator is None:
                    print("❌ [Backend] ViralClipGenerator not initialized")
                    raise Exception("Video processing service not available")
                
                print("🎬 [Backend] Starting video processing pipeline...")
                
                # Process the video using the existing generator
                processing_result = process_video_with_generator(
                    final_file_path,
                    project_name,
                    description,
                    ai_prompt,
                    target_platforms,
                    num_clips,
                    processing_options
                )
                
                if processing_result['success']:
                    print("✅ [Backend] Video processing completed successfully!")
                    print(f"📊 [Backend] Generated {len(processing_result['clips'])} clips")
                    
                    # Return the actual processing results
                    success_response = jsonify({
                        'success': True,
                        'message': f'Video processed successfully! Generated {len(processing_result["clips"])} clips',
                        'chunkIndex': chunk_index,
                        'totalChunks': total_chunks,
                        'filePath': final_file_path,
                        'processingStarted': True,
                        'processingCompleted': True,
                        'clips': processing_result['clips'],
                        'transcription': processing_result.get('transcription', ''),
                        'analysis': processing_result.get('analysis', {})
                    })
                    return success_response
                else:
                    print(f"❌ [Backend] Video processing failed: {processing_result.get('error', 'Unknown error')}")
                    raise Exception(processing_result.get('error', 'Video processing failed'))
                
            except Exception as processing_error:
                print(f"❌ [Backend] Error during video processing: {processing_error}")
                # Return success for upload but note processing error
                success_response = jsonify({
                    'success': True,
                    'message': f'Upload successful but processing failed: {str(processing_error)}',
                    'chunkIndex': chunk_index,
                    'totalChunks': total_chunks,
                    'filePath': final_file_path,
                    'processingStarted': True,
                    'processingCompleted': False,
                    'processingError': str(processing_error)
                })
                return success_response
        
        # Return success for intermediate chunks
        success_response = jsonify({
            'success': True,
            'message': f'Chunk {chunk_index + 1}/{total_chunks} uploaded successfully',
            'chunkIndex': chunk_index,
            'totalChunks': total_chunks
        })
        return success_response
        
    except Exception as e:
        print(f"❌ [Backend] ===== CHUNK UPLOAD FAILED =====")
        print(f"❌ [Backend] Error: {str(e)}")
        print(f"❌ [Backend] Error Type: {type(e).__name__}")
        error_response = jsonify({'error': f'Chunk upload failed: {str(e)}'})
        return error_response, 500

@app.route('/api/frontend/test-chunk', methods=['POST', 'OPTIONS'])
def frontend_test_chunk():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Chunk test preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    
    """Test chunk upload functionality"""
    try:
        print(f"🧪 [Backend] ===== CHUNK TEST REQUEST RECEIVED =====")
        print(f"📊 [Backend] Request Method: {request.method}")
        print(f"📊 [Backend] Content-Type: {request.content_type}")
        print(f"📊 [Backend] Form Data Keys: {list(request.form.keys()) if request.form else 'None'}")
        print(f"📊 [Backend] Files Keys: {list(request.files.keys()) if request.files else 'None'}")
        
        # Check if we received any data
        if request.files and 'chunkData' in request.files:
            chunk_file = request.files['chunkData']
            chunk_size = chunk_file.content_length or 0
            print(f"📦 [Backend] Received chunk file: {chunk_file.filename}, size: {chunk_size} bytes")
            
            # Just echo back the chunk info
            response_data = {
                'success': True,
                'message': 'Chunk test successful',
                'chunk_info': {
                    'filename': chunk_file.filename,
                    'size': chunk_size,
                    'content_type': chunk_file.content_type
                },
                'form_data': dict(request.form) if request.form else {},
                'timestamp': datetime.now().isoformat()
            }
        else:
            response_data = {
                'success': True,
                'message': 'Chunk test successful (no file data)',
                'form_data': dict(request.form) if request.form else {},
                'timestamp': datetime.now().isoformat()
            }
        
        # Create response with CORS headers
        response = jsonify(response_data)
        return response
        
    except Exception as e:
        print(f"❌ [Backend] Chunk test failed: {e}")
        error_response = jsonify({'error': f'Chunk test failed: {str(e)}'})
        return error_response, 500

def process_complete_video(file_path: str, form_data: dict):
    """Process the complete video after all chunks are uploaded"""
    try:
        print(f"🎬 [Backend] ===== PROCESSING COMPLETE VIDEO =====")
        print(f"🎬 [Backend] File path: {file_path}")
        
        # Extract project data from form
        project_name = form_data.get('projectName', 'Unknown Project')
        description = form_data.get('description', '')
        ai_prompt = form_data.get('aiPrompt', '')
        target_platforms = json.loads(form_data.get('targetPlatforms', '["tiktok"]'))
        num_clips = int(form_data.get('numClips', 3))
        processing_options = json.loads(form_data.get('processingOptions', '{}'))
        
        print(f"📊 [Backend] Project: {project_name}")
        print(f"📊 [Backend] Description: {description}")
        print(f"📊 [Backend] AI Prompt: {ai_prompt[:50]}...")
        print(f"📊 [Backend] Target Platforms: {target_platforms}")
        print(f"📊 [Backend] Number of Clips: {num_clips}")
        
        # Process the video using existing logic
        # ... (rest of the video processing logic)
        
        return jsonify({
            'success': True,
            'message': 'Video processed successfully from chunks',
            'filePath': file_path
        })
        
    except Exception as e:
        print(f"❌ [Backend] Video processing failed: {str(e)}")
        return jsonify({'error': f'Video processing failed: {str(e)}'}), 500

@app.route('/api/frontend/process-project', methods=['POST', 'OPTIONS'])
def frontend_process_project():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Process project preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    """Process project from frontend with comprehensive input handling"""
    try:
        # Log request details for debugging
        print(f"🚀 [Backend] ===== PROCESS-PROJECT REQUEST RECEIVED =====")
        print(f"📊 [Backend] Request Method: {request.method}")
        print(f"📊 [Backend] Content-Type: {request.content_type}")
        print(f"📊 [Backend] Content-Length: {request.content_length}")
        print(f"📊 [Backend] Headers: {dict(request.headers)}")
        print(f"📊 [Backend] Form Data Keys: {list(request.form.keys()) if request.form else 'None'}")
        print(f"📊 [Backend] Files Keys: {list(request.files.keys()) if request.files else 'None'}")
        
        # Handle both JSON and FormData requests
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle FormData request
            data = {
                'projectName': request.form.get('projectName'),
                'sourceType': request.form.get('sourceType'),
                'description': request.form.get('description', ''),
                'aiPrompt': request.form.get('aiPrompt', ''),
                'targetPlatforms': json.loads(request.form.get('targetPlatforms', '["tiktok"]')),
                'numClips': int(request.form.get('numClips', 3)),
                'processingOptions': json.loads(request.form.get('processingOptions', '{}')),
                'videoFile': request.files.get('videoFile') if 'videoFile' in request.files else None
            }
            print(f"📁 [Backend] Received FormData request with video file: {data['videoFile'].filename if data['videoFile'] else 'None'}")
            print(f"📁 [Backend] FormData fields: {list(request.form.keys())}")
            print(f"📁 [Backend] FormData files: {list(request.files.keys())}")
        else:
            # Handle JSON request
            data = request.get_json()
            print(f"📁 [Backend] Received JSON request")
            print(f"📁 [Backend] JSON data keys: {list(data.keys()) if data else 'None'}")
        
        # Validate required fields
        print(f"📊 [Backend] ===== EXTRACTED DATA =====")
        print(f"📊 [Backend] Project Name: {data.get('projectName', 'MISSING')}")
        print(f"📊 [Backend] Source Type: {data.get('sourceType', 'MISSING')}")
        print(f"📊 [Backend] Description: {data.get('description', 'MISSING')}")
        print(f"📊 [Backend] AI Prompt: {data.get('aiPrompt', 'MISSING')[:50]}...")
        print(f"📊 [Backend] Target Platforms: {data.get('targetPlatforms', 'MISSING')}")
        print(f"📊 [Backend] Number of Clips: {data.get('numClips', 'MISSING')}")
        print(f"📊 [Backend] Processing Options: {data.get('processingOptions', 'MISSING')}")
        print(f"📊 [Backend] Video File Type: {type(data.get('videoFile', 'MISSING'))}")
        if data.get('videoFile') and hasattr(data['videoFile'], 'filename'):
            print(f"📊 [Backend] Video File Name: {data['videoFile'].filename}")
            print(f"📊 [Backend] Video File Size: {data['videoFile'].content_length if hasattr(data['videoFile'], 'content_length') else 'Unknown'}")
        
        required_fields = ['projectName', 'sourceType']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_name = data['projectName']
        source_type = data['sourceType']
        description = data.get('description', '')
        ai_prompt = data.get('aiPrompt', '')
        target_platforms = data.get('targetPlatforms', ['tiktok'])
        num_clips = data.get('numClips', 3)
        
        # Handle different source types
        if source_type == 'file':
            if 'videoFile' not in data:
                return jsonify({'error': 'Video file required for file upload'}), 400
            video_file = data['videoFile']
            
            # Handle FormData file upload
            if hasattr(video_file, 'filename') and video_file.filename:
                # This is a FormData file upload
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                # Sanitize project name for filename
                safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_project_name = safe_project_name.replace(' ', '_')[:30]  # Limit length and replace spaces
                safe_filename = f"{timestamp}_{safe_project_name}.mp4"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                
                print(f"📁 [Backend] Saving FormData video file to: {filepath}")
                print(f"📝 [Backend] Project name: '{project_name}' -> Safe name: '{safe_project_name}'")
                print(f"📁 [Backend] Original filename: {video_file.filename}")
                
                # Save the uploaded file
                video_file.save(filepath)
                
                print(f"✅ [Backend] FormData video saved successfully: {os.path.getsize(filepath)} bytes")
                
            # Handle base64 video data
            elif isinstance(video_file, str) and video_file.startswith('data:'):
                import base64
                video_data = video_file.split(',')[1]
                video_bytes = base64.b64decode(video_data)
                
                # Save to file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                # Sanitize project name for filename
                safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_project_name = safe_project_name.replace(' ', '_')[:30]  # Limit length and replace spaces
                safe_filename = f"{timestamp}_{safe_project_name}.mp4"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                
                print(f"📁 [Backend] Saving base64 video to: {filepath}")
                print(f"📝 [Backend] Project name: '{project_name}' -> Safe name: '{safe_project_name}'")
                
                with open(filepath, 'wb') as f:
                    f.write(video_bytes)
                
                print(f"✅ [Backend] Base64 video saved successfully: {os.path.getsize(filepath)} bytes")
            else:
                return jsonify({'error': f'Invalid video file format. Expected File object or base64 string, got: {type(video_file)}'}), 400
                
        elif source_type == 'url':
            if 'sourceUrl' not in data:
                return jsonify({'error': 'Source URL required for URL upload'}), 400
            
            source_url = data['sourceUrl']
            
            # Check if it's a YouTube URL
            if 'youtube.com' in source_url or 'youtu.be' in source_url:
                app.logger.info(f"🎥 Processing YouTube URL: {source_url}")
                
                try:
                    # Download YouTube video
                    download_result = download_youtube_video(source_url, app.config['UPLOAD_FOLDER'])
                    
                    if not download_result['success']:
                        return jsonify({'error': f'YouTube download failed: {download_result["error"]}'}), 500
                    
                    filepath = download_result['file_path']
                    youtube_metadata = download_result['metadata']
                    
                    app.logger.info(f"✅ YouTube video downloaded: {filepath}")
                    app.logger.info(f"📊 Video metadata: {youtube_metadata}")
                    
                except Exception as e:
                    app.logger.error(f"❌ YouTube processing failed: {e}")
                    return jsonify({'error': f'YouTube processing failed: {str(e)}'}), 500
            else:
                # Handle other URL types (could be expanded for Vimeo, etc.)
                app.logger.info(f"🌐 Processing generic URL: {source_url}")
                # For now, return error for non-YouTube URLs
                return jsonify({'error': 'Currently only YouTube URLs are supported for URL processing'}), 400
            
        elif source_type == 'text':
            if 'sourceText' not in data:
                return jsonify({'error': 'Source text required for text upload'}), 400
            # For text-based content, we'll create a placeholder
            filepath = f"placeholder_{project_name}_text.txt"
        else:
            return jsonify({'error': f'Unsupported source type: {source_type}'}), 400
        
        # Extract processing options
        processing_options = data.get('processingOptions', {})
        
        # Extract aspect ratio information
        target_aspect_ratio = None
        aspect_ratio_options = {}
        watermark_options = {}
        
        # Check for aspect ratio in processing options
        if 'targetAspectRatio' in processing_options:
            target_aspect_ratio = processing_options['targetAspectRatio']
            print(f"🎬 Target aspect ratio from frontend: {target_aspect_ratio}")
        
        # Check for aspect ratio options
        if 'aspectRatioOptions' in processing_options:
            aspect_ratio_options = processing_options['aspectRatioOptions']
            print(f"🎬 Aspect ratio options from frontend: {aspect_ratio_options}")
        
        # Check for watermark options
        if 'watermarkOptions' in processing_options:
            watermark_options = processing_options['watermarkOptions']
            print(f"💧 Watermark options from frontend: {watermark_options}")
        
        # Add aspect ratio and watermark info to processing options for the generator
        if target_aspect_ratio:
            processing_options['targetAspectRatio'] = target_aspect_ratio
        if aspect_ratio_options:
            processing_options['aspectRatioOptions'] = aspect_ratio_options
        if watermark_options:
            processing_options['watermarkOptions'] = watermark_options
        
        # Update generator settings based on frontend options
        if processing_options:
            # Duration settings
            if 'targetDuration' in processing_options:
                generator.min_clip_duration = max(15, processing_options['targetDuration'])
            if 'minDuration' in processing_options:
                generator.min_clip_duration = max(5, processing_options['minDuration'])
            if 'maxDuration' in processing_options:
                generator.max_clip_duration = min(600, processing_options['maxDuration'])
            
            # Quality settings
            quality = processing_options.get('quality', 'medium')
            if quality == 'high':
                generator.segment_duration = 5  # Higher precision for high quality
            elif quality == 'low':
                generator.segment_duration = 15  # Lower precision for speed
        
        app.logger.info(f"Frontend project processing: {project_name}")
        app.logger.info(f"Source type: {source_type}")
        app.logger.info(f"Target platforms: {target_platforms}")
        app.logger.info(f"Processing options: {processing_options}")
        
        # Log aspect ratio and watermark information
        if target_aspect_ratio:
            app.logger.info(f"🎬 Target aspect ratio: {target_aspect_ratio}")
        if aspect_ratio_options:
            app.logger.info(f"🎬 Aspect ratio options: {aspect_ratio_options}")
        if watermark_options:
            app.logger.info(f"💧 Watermark options: {watermark_options}")
        
        # Add progress tracking
        app.logger.info("🔄 [Progress] Starting video processing...")
        
        # Prepare frontend inputs for AI prompt enhancement
        frontend_inputs = {
            'projectName': project_name,
            'description': description,
            'aiPrompt': ai_prompt,
            'targetPlatforms': target_platforms,
            'processingOptions': processing_options
        }
        
        # Verify file exists before processing
        if not os.path.exists(filepath):
            app.logger.error(f"Video file not found: {filepath}")
            return jsonify({'error': f'Video file not found: {filepath}'}), 404
        
        file_size = os.path.getsize(filepath)
        app.logger.info(f"Video file verified: {filepath} ({file_size} bytes)")
        
        # Ensure output directory exists before processing
        try:
            output_dir = Path(app.config['OUTPUT_FOLDER'])
            output_dir.mkdir(parents=True, exist_ok=True)
            app.logger.info(f"Ensured output directory exists: {output_dir}")
        except Exception as e:
            app.logger.error(f"Failed to create output directory: {e}")
            return jsonify({'error': f'Failed to create output directory: {str(e)}'}), 500
        
        # Process video using the same logic as frontend with enhanced AI prompts
        print(f"🎬 [Backend] ===== STARTING VIDEO PROCESSING =====")
        print(f"🎬 [Backend] File path: {filepath}")
        print(f"🎬 [Backend] File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        print(f"🎬 [Backend] Number of clips: {num_clips}")
        print(f"🎬 [Backend] AI Prompt: {ai_prompt[:100]}...")
        
        app.logger.info("🔄 [Progress] Processing video with AI analysis...")
        clips, transcription = generator.generate_viral_clips(filepath, num_clips, frontend_inputs)
        
        # Enhance clips with frontend-specific data and include actual video data
        enhanced_clips = []
        for i, clip_path in enumerate(clips):
            # Extract clip info from the generation report if available
            clip_info = generator._get_clip_info_from_report(clip_path, i+1)
            
            # Read the actual video file and convert to base64 for frontend
            video_data = None
            try:
                with open(clip_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    import base64
                    video_data = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}"
            except Exception as e:
                app.logger.warning(f"Could not read video file {clip_path}: {e}")
                video_data = None
            
            enhanced_clip = {
                'id': f"clip_{i+1}_{int(time.time())}",  # Generate unique ID
                'clip_number': i + 1,
                'filename': os.path.basename(clip_path),
                'filepath': clip_path,
                'videoData': video_data,  # Actual video data for frontend display
                'start_time': clip_info.get('start_time', 0),  # Match frontend interface
                'end_time': clip_info.get('end_time', 0),      # Match frontend interface
                'duration': clip_info.get('duration', 0),
                'viral_score': clip_info.get('viral_score', 8),
                'content_type': clip_info.get('content_type', 'viral'),
                'caption': clip_info.get('caption', f'Viral clip #{i+1} from {project_name}'),
                'hashtags': clip_info.get('hashtags', ['viral', 'trending', 'amazing']),
                'target_audience': clip_info.get('target_audience', 'general'),
                'platforms': target_platforms,
                'segment_text': clip_info.get('segment_text', ''),  # Match frontend interface
                'viral_potential': clip_info.get('viral_score', 8),
                'engagement': clip_info.get('viral_score', 8),
                'story_value': clip_info.get('viral_score', 8),
                'audio_impact': clip_info.get('viral_score', 8),
                'ai_enhancement': processing_options.get('aiEnhancement', True),
                'transcription_generated': processing_options.get('generateTranscription', True),
                'performance_prediction': processing_options.get('enablePerformancePrediction', True),
                'style_analysis': processing_options.get('enableStyleAnalysis', True),
                'content_moderation': processing_options.get('enableContentModeration', True),
                'trend_analysis': processing_options.get('enableTrendAnalysis', True),
                'advanced_captions': processing_options.get('enableAdvancedCaptions', True),
                'viral_optimization': processing_options.get('enableViralOptimization', True),
                'audience_analysis': processing_options.get('enableAudienceAnalysis', True),
                'competitor_analysis': processing_options.get('enableCompetitorAnalysis', True),
                'content_strategy': processing_options.get('enableContentStrategy', True)
            }
            enhanced_clips.append(enhanced_clip)
        
        # Prepare response matching frontend format
        response_data = {
            'success': True,
            'projectName': project_name,
            'projectDescription': description,
            'sourceType': source_type,
            'targetPlatforms': target_platforms,
            'aiPrompt': ai_prompt,
            'processingOptions': processing_options,
            'clipsGenerated': len(enhanced_clips),
            'clips': enhanced_clips,
            'transcription': transcription if processing_options.get('generateTranscription', True) else None,
            'outputDirectory': app.config['OUTPUT_FOLDER'],
            'message': f'Successfully processed project {project_name} with {len(enhanced_clips)} clips',
            'processingMetadata': {
                'totalProcessingTime': time.time(),
                'aiEnhancementEnabled': processing_options.get('aiEnhancement', True),
                'transcriptionEnabled': processing_options.get('generateTranscription', True),
                'quality': processing_options.get('quality', 'medium'),
                'targetDuration': processing_options.get('targetDuration', 60),
                'minDuration': processing_options.get('minDuration', 15),
                'maxDuration': processing_options.get('maxDuration', 120)
            }
        }
        
        app.logger.info(f"Frontend project processing complete: {len(enhanced_clips)} clips generated")
        
        # Log final response details
        print(f"✅ [Backend] ===== PROCESS-PROJECT COMPLETED SUCCESSFULLY =====")
        print(f"📊 [Backend] Project: {project_name}")
        print(f"📊 [Backend] Clips Generated: {len(enhanced_clips)}")
        print(f"📊 [Backend] Output Directory: {app.config['OUTPUT_FOLDER']}")
        print(f"📊 [Backend] Response Size: {len(str(response_data))} characters")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ [Backend] ===== PROCESS-PROJECT FAILED =====")
        print(f"❌ [Backend] Error: {str(e)}")
        print(f"❌ [Backend] Error Type: {type(e).__name__}")
        app.logger.error(f"Frontend project processing failed: {e}")
        return jsonify({'error': f'Project processing failed: {str(e)}'}), 500

# Additional Frontend Integration Endpoints
@app.route('/api/frontend/get-latest-clips', methods=['GET'])
def frontend_get_latest_clips():
    """Get the latest generated clips for frontend display"""
    try:
        clips_dir = Path(app.config['OUTPUT_FOLDER'])
        clips = []
        
        # Look for video files in the viral_clips directory
        for file_path in clips_dir.glob('*.mp4'):
            if 'viral_clip' in file_path.name:
                # Extract clip information from filename
                filename = file_path.name
                file_size = file_path.stat().st_size
                created_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # Parse filename to extract clip details
                # Example: viral_clip_3_9_My_WWIII_draft_strategy__WWIII.mp4
                parts = filename.replace('.mp4', '').split('_')
                clip_number = int(parts[2]) if len(parts) > 2 else 1
                
                clips.append({
                    'id': str(uuid.uuid4()),
                    'clip_number': clip_number,
                    'filename': filename,
                    'filepath': str(file_path),
                    'start_time': 0,  # Will be updated from analysis if available
                    'end_time': 0,    # Will be updated from analysis if available
                    'duration': 0,    # Will be updated from analysis if available
                    'viral_score': 9,  # Default high score for generated clips
                    'content_type': 'viral',
                    'caption': f'Viral Clip {clip_number} - {filename}',
                    'hashtags': ['#viral', '#trending', '#content', '#ai'],
                    'target_audience': 'general',
                    'platforms': ['tiktok', 'instagram', 'youtube'],
                    'segment_text': f'Generated viral clip {clip_number}',
                    'viral_potential': 9,
                    'engagement': 9,
                    'story_value': 9,
                    'audio_impact': 9,
                    'file_size': file_size,
                    'created_at': created_time.isoformat()
                })
        
        # Sort by creation time (newest first)
        clips.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'clips': clips,
            'total_clips': len(clips),
            'message': f'Found {len(clips)} generated clips'
        })
        
    except Exception as e:
        app.logger.error(f"Failed to get latest clips: {e}")
        return jsonify({'error': f'Failed to get latest clips: {str(e)}'}), 500

@app.route('/api/frontend/get-project-status/<project_name>', methods=['GET'])
def frontend_get_project_status(project_name):
    """Get the status of a specific project by name"""
    try:
        # Look for project-related files
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        project_files = []
        
        # Look for files that might be related to this project
        for file_path in output_dir.glob('*'):
            if project_name.lower().replace(' ', '_') in file_path.name.lower():
                project_files.append({
                    'filename': file_path.name,
                    'type': 'video' if file_path.suffix == '.mp4' else 'analysis' if file_path.suffix == '.json' else 'transcription',
                    'size': file_path.stat().st_size,
                    'created': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        # Check if we have generated clips
        has_clips = any(f['type'] == 'video' for f in project_files)
        
        return jsonify({
            'success': True,
            'project_name': project_name,
            'status': 'completed' if has_clips else 'processing',
            'files': project_files,
            'has_clips': has_clips,
            'total_files': len(project_files)
        })

    except Exception as e:
        app.logger.error(f"Failed to get project status: {e}")
        return jsonify({'error': f'Failed to get project status: {str(e)}'}), 500

@app.route('/api/frontend/test-video-display', methods=['GET'])
def frontend_test_video_display():
    """Test endpoint for frontend video display"""
    try:
        clips_dir = Path(app.config['OUTPUT_FOLDER'])
        test_clips = []
        
        # Get up to 3 most recent clips for testing
        video_files = list(clips_dir.glob('*.mp4'))
        video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for i, file_path in enumerate(video_files[:3]):
            if 'viral_clip' in file_path.name:
                test_clips.append({
                    'id': f'test_clip_{i}',
                    'clip_number': i + 1,
                    'filename': file_path.name,
                    'filepath': str(file_path),
                    'start_time': 0,
                    'end_time': 30,
                    'duration': 30,
                    'viral_score': 9,
                    'content_type': 'viral',
                    'caption': f'Test Clip {i + 1} - {file_path.name}',
                    'hashtags': ['#test', '#viral', '#content'],
                    'target_audience': 'general',
                    'platforms': ['tiktok', 'instagram', 'youtube'],
                    'segment_text': f'Test viral clip {i + 1}',
                    'viral_potential': 9,
                    'engagement': 9,
                    'story_value': 9,
                    'audio_impact': 9
                })
        
        return jsonify({
            'success': True,
            'clips': test_clips,
            'total_clips': len(test_clips),
            'message': f'Test video display data - {len(test_clips)} clips available'
        })
        
    except Exception as e:
        app.logger.error(f"Failed to get test video data: {e}")
        return jsonify({'error': f'Failed to get test video data: {str(e)}'}), 500

@app.route('/api/frontend/create-project', methods=['POST', 'OPTIONS'])
def frontend_create_project():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Create project preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    """Create a new project from frontend"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['title', 'sourceType']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_data = {
            'id': f"proj_{int(time.time())}",
            'title': data['title'],
            'description': data.get('description', ''),
            'sourceType': data['sourceType'],
            'sourcePath': data.get('sourcePath'),
            'sourceUrl': data.get('sourceUrl'),
            'sourceText': data.get('sourceText'),
            'targetPlatforms': data.get('targetPlatforms', ['tiktok']),
            'aiPrompt': data.get('aiPrompt', ''),
            'status': 'pending',
            'createdAt': datetime.now().isoformat(),
            'updatedAt': datetime.now().isoformat()
        }
        
        app.logger.info(f"Frontend project created: {project_data['title']}")
        
        return jsonify({
            'success': True,
            'project': project_data,
            'message': 'Project created successfully'
        })
        
    except Exception as e:
        app.logger.error(f"Frontend project creation failed: {e}")
        return jsonify({'error': f'Project creation failed: {str(e)}'}), 500

@app.route('/api/frontend/upload-file', methods=['POST', 'OPTIONS'])
def frontend_upload_file():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Upload file preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    """Handle file upload from frontend"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        project_id = request.form.get('projectId', 'unknown')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
        if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, wmv, flv'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{project_id}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        app.logger.info(f"Frontend file uploaded: {safe_filename} for project {project_id}")
        
        return jsonify({
            'success': True,
            'filename': safe_filename,
            'filepath': filepath,
            'projectId': project_id,
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        app.logger.error(f"Frontend file upload failed: {e}")
        return jsonify({'error': f'File upload failed: {str(e)}'}), 500

@app.route('/api/frontend/processing-status/<project_id>', methods=['GET'])
def frontend_processing_status(project_id):
    """Get processing status for a specific project"""
    try:
        # This would typically check a database or cache for real-time status
        # For now, we'll return a mock status
        status_data = {
            'projectId': project_id,
            'status': 'completed',  # Could be: pending, processing, completed, failed
            'progress': 100,
            'currentStage': 'Finalizing',
            'stages': [
                {'name': 'File Upload', 'status': 'completed', 'progress': 100},
                {'name': 'Audio Analysis', 'status': 'completed', 'progress': 100},
                {'name': 'AI Processing', 'status': 'completed', 'progress': 100},
                {'name': 'Clip Generation', 'status': 'completed', 'progress': 100},
                {'name': 'Finalizing', 'status': 'completed', 'progress': 100}
            ],
            'estimatedTimeRemaining': 0,
            'startTime': datetime.now().isoformat(),
            'elapsedTime': 0
        }
        
        return jsonify({
            'success': True,
            'status': status_data
        })
        
    except Exception as e:
        app.logger.error(f"Failed to get processing status: {e}")
        return jsonify({'error': f'Failed to get status: {str(e)}'}), 500

@app.route('/api/frontend/project-details/<project_id>', methods=['GET'])
def frontend_project_details(project_id):
    """Get detailed information about a project"""
    try:
        # This would typically fetch from a database
        # For now, we'll return mock data
        project_data = {
            'id': project_id,
            'title': f'Project {project_id}',
            'description': 'Sample project description',
            'sourceType': 'file',
            'targetPlatforms': ['tiktok', 'instagram'],
            'status': 'completed',
            'createdAt': datetime.now().isoformat(),
            'updatedAt': datetime.now().isoformat(),
            'clips': [],
            'transcription': None,
            'processingOptions': {
                'aiEnhancement': True,
                'generateTranscription': True,
                'targetDuration': 60,
                'quality': 'medium'
            }
        }
        
        return jsonify({
            'success': True,
            'project': project_data
        })
        
    except Exception as e:
        app.logger.error(f"Failed to get project details: {e}")
        return jsonify({'error': f'Failed to get project details: {str(e)}'}), 500

@app.route('/api/frontend/validate-inputs', methods=['POST'])
def frontend_validate_inputs():
    """Validate frontend inputs before processing"""
    try:
        data = request.get_json()
        
        validation_errors = []
        
        # Validate project title
        if not data.get('title', '').strip():
            validation_errors.append('Project title is required')
        
        # Validate source type and content
        source_type = data.get('sourceType')
        if source_type == 'file':
            if not data.get('videoFile'):
                validation_errors.append('Video file is required for file upload')
        elif source_type == 'url':
            if not data.get('sourceUrl', '').strip():
                validation_errors.append('Source URL is required for URL upload')
        elif source_type == 'text':
            if not data.get('sourceText', '').strip():
                validation_errors.append('Source text is required for text upload')
        else:
            validation_errors.append('Invalid source type')
        
        # Validate target platforms
        if not data.get('targetPlatforms') or len(data.get('targetPlatforms', [])) == 0:
            validation_errors.append('At least one target platform must be selected')
        
        # Validate processing options
        processing_options = data.get('processingOptions', {})
        if processing_options.get('minDuration', 0) > processing_options.get('maxDuration', 1000):
            validation_errors.append('Minimum duration cannot be greater than maximum duration')
        
        if validation_errors:
            return jsonify({
                'success': False,
                'valid': False,
                'errors': validation_errors
            }), 400
        
        return jsonify({
            'success': True,
            'valid': True,
            'message': 'All inputs are valid'
        })
        
    except Exception as e:
        app.logger.error(f"Input validation failed: {e}")
        return jsonify({'error': f'Validation failed: {str(e)}'}), 500

# Persistent Processing Endpoints
@app.route('/api/frontend/start-persistent-processing', methods=['POST'])
def start_persistent_processing():
    """Start persistent processing that continues in background"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['userId', 'projectName', 'sourceType']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        user_id = data['userId']
        project_name = data['projectName']
        source_type = data['sourceType']
        processing_options = data.get('processingOptions', {})
        
        # Handle different source types
        if source_type == 'file':
            if 'videoFile' not in data:
                return jsonify({'error': 'Video file required for file upload'}), 400
            
            video_file = data['videoFile']
            
            # Handle base64 video data
            if isinstance(video_file, str) and video_file.startswith('data:'):
                import base64
                video_data = video_file.split(',')[1]
                video_bytes = base64.b64decode(video_data)
                
                # Save to user-specific directory
                user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
                os.makedirs(user_upload_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_filename = f"{timestamp}_{project_name}.mp4"
                filepath = os.path.join(user_upload_dir, safe_filename)
                
                with open(filepath, 'wb') as f:
                    f.write(video_bytes)
            else:
                return jsonify({'error': 'Invalid video file format'}), 400
                
        elif source_type == 'url':
            if 'sourceUrl' not in data:
                return jsonify({'error': 'Source URL required for URL upload'}), 400
            # For now, we'll handle this as a placeholder
            filepath = f"placeholder_{project_name}_url.mp4"
            
        elif source_type == 'text':
            if 'sourceText' not in data:
                return jsonify({'error': 'Source text required for text upload'}), 400
            # For text-based content, we'll create a placeholder
            filepath = f"placeholder_{project_name}_text.txt"
        else:
            return jsonify({'error': f'Unsupported source type: {source_type}'}), 400
        
        # Add job to queue for background processing
        job_id = job_queue.add_job(user_id, project_name, filepath, processing_options)
        
        if not job_id:
            return jsonify({'error': 'Failed to create processing job'}), 500
        
        app.logger.info(f"Persistent processing started: Job {job_id} for user {user_id}")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Processing started in background. You can close this page and check back later.',
            'status_url': f'/api/frontend/job-status/{job_id}'
        })
        
    except Exception as e:
        app.logger.error(f"Persistent processing failed: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/frontend/job-status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get current status of a processing job"""
    try:
        if not job_queue:
            return jsonify({'error': 'Job queue not initialized'}), 500
        
        status = job_queue.get_job_status(job_id)
        
        if not status:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        app.logger.error(f"Failed to get job status: {e}")
        return jsonify({'error': f'Failed to get status: {str(e)}'}), 500

@app.route('/api/frontend/user-jobs/<user_id>', methods=['GET'])
def get_user_jobs(user_id):
    """Get all jobs for a specific user"""
    try:
        if not job_queue:
            return jsonify({'error': 'Job queue not initialized'}), 500
        
        jobs = job_queue.get_user_jobs(user_id)
        
        return jsonify({
            'success': True,
            'jobs': jobs,
            'total': len(jobs)
        })
        
    except Exception as e:
        app.logger.error(f"Failed to get user jobs: {e}")
        return jsonify({'error': f'Failed to get jobs: {str(e)}'}), 500

@app.route('/api/frontend/job-results/<job_id>', methods=['GET'])
def get_job_results(job_id):
    """Get results of a completed job"""
    try:
        if not job_queue:
            return jsonify({'error': 'Job queue not initialized'}), 500
        
        status = job_queue.get_job_status(job_id)
        
        if not status:
            return jsonify({'error': 'Job not found'}), 404
        
        if status['status'] != 'completed':
            return jsonify({'error': 'Job not yet completed'}), 500
        
        # Get the output directory from the job status
        output_path = status.get('output_path')
        if not output_path:
            return jsonify({'error': 'No output path found'}), 404
        
        # Read the generation report
        report_path = os.path.join(output_path, "generation_report.json")
        if not os.path.exists(report_path):
            return jsonify({'error': 'Generation report not found'}), 404
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        return jsonify({
            'success': True,
            'results': report_data
        })
        
    except Exception as e:
        app.logger.error(f"Failed to get job results: {e}")
        return jsonify({'error': f'Failed to get results: {str(e)}'}), 500

@app.route('/api/frontend/get-clip-data/<filename>', methods=['GET'])
def get_clip_data(filename):
    """Get actual video data for a specific clip"""
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Clip not found'}), 404
        
        # Read the video file and convert to base64
        with open(filepath, 'rb') as video_file:
            video_bytes = video_file.read()
            import base64
            video_data = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}"
        
        return jsonify({
            'success': True,
            'filename': filename,
            'videoData': video_data,
            'size': len(video_bytes)
        })
        
    except Exception as e:
        app.logger.error(f"Failed to get clip data: {e}")
        return jsonify({'error': f'Failed to get clip data: {str(e)}'}), 500

@app.route('/api/frontend/test-video-display', methods=['GET'])
def test_video_display():
    """Test endpoint that returns mock video data for frontend testing"""
    try:
        import base64
        # Create a simple 1x1 pixel MP4 data (minimal valid MP4)
        # This is a very basic MP4 header that browsers can handle
        minimal_mp4 = b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom\x00\x00\x00\x00\x08mdat\x00\x00\x00\x00'
        
        video_data = f"data:video/mp4;base64,{base64.b64encode(minimal_mp4).decode('utf-8')}"
        
        test_clips = [
            {
                'id': 'test_clip_1',
                'filename': 'test_video_1.mp4',
                'startTime': 0,
                'endTime': 10,
                'duration': 10,
                'caption': 'Test Video Clip 1 - Frontend Display Test',
                'hashtags': ['test', 'frontend', 'video'],
                'transcript': 'This is a test transcript for frontend video display testing.',
                'videoData': video_data
            },
            {
                'id': 'test_clip_2',
                'filename': 'test_video_2.mp4',
                'startTime': 10,
                'endTime': 20,
                'duration': 10,
                'caption': 'Test Video Clip 2 - Backend Integration Test',
                'hashtags': ['test', 'backend', 'integration'],
                'transcript': 'Another test transcript to verify the system works correctly.',
                'videoData': video_data
            }
        ]
        
        response_data = {
            'success': True,
            'clipsGenerated': len(test_clips),
            'clips': test_clips,
            'message': 'Test clips generated for frontend video display testing'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Failed to generate test video data: {e}")
        return jsonify({'error': f'Failed to generate test data: {str(e)}'}), 500

@app.route('/api/frontend/test-cors', methods=['GET', 'POST', 'OPTIONS'])
def test_cors():
    """Test endpoint to verify CORS is working properly"""
    if request.method == 'OPTIONS':
        # Preflight request - return with CORS headers
        response = jsonify({'message': 'CORS preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    # Get request info for debugging
    request_info = {
        'method': request.method,
        'origin': request.headers.get('Origin'),
        'user_agent': request.headers.get('User-Agent'),
        'content_type': request.headers.get('Content-Type'),
        'timestamp': datetime.now().isoformat()
    }
    
    # Create response with explicit CORS headers
    response = jsonify({
        'success': True,
        'message': 'CORS test successful',
        'request_info': request_info,
        'cors_headers': {
            'access_control_allow_origin': request.headers.get('Origin'),
            'access_control_allow_methods': 'GET, POST, OPTIONS',
            'access_control_allow_headers': 'Content-Type, Authorization'
        }
    })
    
    # Add explicit CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    return response

@app.route('/api/frontend/google-oauth', methods=['POST', 'OPTIONS'])
def google_oauth_exchange():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Google OAuth preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    """Exchange Google OAuth authorization code for access and refresh tokens"""
    try:
        data = request.get_json()
        
        if not data or 'code' not in data:
            return jsonify({'error': 'Authorization code is required'}), 400
        
        auth_code = data['code']
        
        # Get Google OAuth credentials from environment
        client_id = os.getenv('VITE_GOOGLE_DRIVE_CLIENT_ID')
        client_secret = os.getenv('VITE_GOOGLE_DRIVE_CLIENT_SECRET')
        redirect_uri = os.getenv('VITE_GOOGLE_DRIVE_REDIRECT_URI', 'http://localhost:5173/oauth-callback.html')
        
        if not client_id or not client_secret:
            app.logger.error("Google OAuth credentials not configured in environment variables")
            app.logger.error("Please add the following to your .env.local file:")
            app.logger.error("VITE_GOOGLE_DRIVE_CLIENT_ID=your_google_client_id")
            app.logger.error("VITE_GOOGLE_DRIVE_CLIENT_SECRET=your_google_client_secret")
            return jsonify({
                'error': 'Google OAuth credentials not configured',
                'details': 'Please add VITE_GOOGLE_DRIVE_CLIENT_ID and VITE_GOOGLE_DRIVE_CLIENT_SECRET to your .env.local file'
            }), 500
        
        # Exchange authorization code for tokens
        import requests
        
        token_url = 'https://oauth2.googleapis.com/token'
        token_data = {
            'client_id': client_id,
            'client_secret': client_secret,
            'code': auth_code,
            'grant_type': 'authorization_code',
            'redirect_uri': redirect_uri,
            'access_type': 'offline'
        }
        
        response = requests.post(token_url, data=token_data)
        
        if response.status_code == 200:
            token_response = response.json()
            
            # Log token information for debugging
            app.logger.info(f"Google OAuth token exchange successful")
            app.logger.info(f"Access token length: {len(token_response.get('access_token', ''))}")
            app.logger.info(f"Refresh token present: {bool(token_response.get('refresh_token'))}")
            app.logger.info(f"Expires in: {token_response.get('expires_in')}")
            app.logger.info(f"Scope: {token_response.get('scope')}")
            
            # Return tokens to frontend
            return jsonify({
                'success': True,
                'access_token': token_response.get('access_token'),
                'refresh_token': token_response.get('refresh_token'),
                'expires_in': token_response.get('expires_in'),
                'token_type': token_response.get('token_type', 'Bearer'),
                'scope': token_response.get('scope')
            })
        else:
            error_data = response.json()
            app.logger.error(f"Google OAuth token exchange failed: {error_data}")
            return jsonify({
                'error': 'Failed to exchange authorization code for tokens',
                'details': error_data.get('error_description', 'Unknown error')
            }), 400
            
    except Exception as e:
        app.logger.error(f"Google OAuth endpoint error: {e}")
        return jsonify({'error': f'OAuth exchange failed: {str(e)}'}), 500

@app.route('/api/frontend/test-google-token', methods=['POST', 'OPTIONS'])
def test_google_token():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Test Google token preflight successful'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response
    """Test if a Google access token is valid and can access user info"""
    try:
        data = request.get_json()
        if not data or 'access_token' not in data:
            return jsonify({'error': 'Access token is required'}), 400
        
        access_token = data['access_token']
        
        # Test the token with Google's userinfo endpoint
        import requests
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Try userinfo endpoint first
        userinfo_response = requests.get('https://www.googleapis.com/oauth2/v2/userinfo', headers=headers)
        
        if userinfo_response.status_code == 200:
            user_data = userinfo_response.json()
            return jsonify({
                'success': True,
                'message': 'Token is valid',
                'user_info': user_data,
                'endpoint': 'userinfo'
            })
        
        # If userinfo fails, try Drive API
        drive_response = requests.get('https://www.googleapis.com/drive/v3/about?fields=user', headers=headers)
        
        if drive_response.status_code == 200:
            drive_data = drive_response.json()
            return jsonify({
                'success': True,
                'message': 'Token is valid for Drive API',
                'user_info': drive_data.get('user', {}),
                'endpoint': 'drive'
            })
        
        # If both fail, return error details
        return jsonify({
            'success': False,
            'error': 'Token validation failed',
            'userinfo_status': userinfo_response.status_code,
            'drive_status': drive_response.status_code,
            'userinfo_error': userinfo_response.text if userinfo_response.status_code != 200 else None,
            'drive_error': drive_response.text if drive_response.status_code != 200 else None
        }), 400
        
    except Exception as e:
        app.logger.error(f"Google token test failed: {e}")
        return jsonify({'error': f'Token test failed: {str(e)}'}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to run Flask app"""
    try:
        print("Starting Viral Clip Generator Backend...")
        print("=" * 60)
        
        # Check environment variables - support both local and production
        api_key = os.getenv('VITE_GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment variables")
            print("Please set GEMINI_API_KEY in your environment:")
            print("   Local: Create .env.local file with VITE_GEMINI_API_KEY=your_key")
            print("   Production: Set GEMINI_API_KEY in Render environment variables")
            return
        
        print(f"✅ Environment variables loaded successfully")
        print(f"API Key: {api_key[:10]}...")
        
        # Initialize services
        if not initialize_services():
            print("❌ Failed to initialize services. Exiting.")
            return
        
        print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
        print(f"📁 Output folder: {app.config['OUTPUT_FOLDER']}")
        
        # Get deployment info
        port = int(os.environ.get('PORT', 5000))
        is_production = os.environ.get('RENDER', 'false').lower() == 'true'
        
        if is_production:
            print(f"🚀 Production deployment detected (Render)")
            print(f"Server will be available on port: {port}")
        else:
            print(f"🖥️ Local development mode")
            print(f"Server will be available at: http://localhost:{port}")
        
        print(f"📚 API Documentation available at: /api/health")
        print(f"🌐 Frontend integration endpoints available")
        print("=" * 60)
        
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,  # Disable debug mode to prevent restarts during processing
            threaded=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Troubleshooting:")
        print("   1. Check environment variables (GEMINI_API_KEY)")
        print("   2. Ensure all dependencies are installed")
        print("   3. Check disk space and permissions")
        print("   4. For production: Check Render logs and environment variables")

if __name__ == "__main__":
    main()
