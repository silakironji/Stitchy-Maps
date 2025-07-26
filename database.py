import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import uuid

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnalysisSession(Base):
    """
    Table to store analysis session information
    """
    __tablename__ = "analysis_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    session_name = Column(String, nullable=True)
    total_images = Column(Integer, default=0)
    stitching_completed = Column(Boolean, default=False)
    ndvi_enabled = Column(Boolean, default=False)
    detector_type = Column(String, default="SIFT")
    match_threshold = Column(Float, default=0.75)
    ransac_threshold = Column(Float, default=5.0)
    band_type = Column(String, default="standard_rgb")
    notes = Column(Text, nullable=True)

class ImageMetadata(Base):
    """
    Table to store individual image metadata
    """
    __tablename__ = "image_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    filename = Column(String, nullable=False)
    file_size = Column(Integer)
    image_width = Column(Integer)
    image_height = Column(Integer)
    upload_time = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    processing_time = Column(Float, nullable=True)  # in seconds

class StitchingResult(Base):
    """
    Table to store image stitching results
    """
    __tablename__ = "stitching_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    result_width = Column(Integer)
    result_height = Column(Integer)
    processing_time = Column(Float)  # in seconds
    success = Column(Boolean)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    settings = Column(JSON)  # Store stitching settings as JSON

class NDVIAnalysis(Base):
    """
    Table to store NDVI analysis results
    """
    __tablename__ = "ndvi_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    image_type = Column(String)  # 'individual' or 'stitched'
    image_filename = Column(String, nullable=True)  # For individual images
    band_configuration = Column(String)
    mean_ndvi = Column(Float)
    std_ndvi = Column(Float)
    min_ndvi = Column(Float)
    max_ndvi = Column(Float)
    vegetation_coverage = Column(Float)
    dense_vegetation_percentage = Column(Float)
    moderate_vegetation_percentage = Column(Float)
    sparse_vegetation_percentage = Column(Float)
    bare_soil_percentage = Column(Float)
    water_percentage = Column(Float)
    processing_time = Column(Float)  # in seconds
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """
    Database manager class for handling all database operations
    """
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            return True
        except SQLAlchemyError as e:
            print(f"Error creating tables: {e}")
            return False
    
    def get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()
    
    def create_analysis_session(self, session_name: Optional[str] = None, 
                              total_images: int = 0, 
                              detector_type: str = "SIFT",
                              match_threshold: float = 0.75,
                              ransac_threshold: float = 5.0,
                              ndvi_enabled: bool = False,
                              band_type: str = "standard_rgb") -> str:
        """
        Create a new analysis session
        
        Returns:
            session_id: Unique identifier for the session
        """
        db = self.get_session()
        try:
            session = AnalysisSession(
                session_name=session_name,
                total_images=total_images,
                detector_type=detector_type,
                match_threshold=match_threshold,
                ransac_threshold=ransac_threshold,
                ndvi_enabled=ndvi_enabled,
                band_type=band_type
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.session_id
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error creating analysis session: {e}")
            return None
        finally:
            db.close()
    
    def update_session(self, session_id: str, **kwargs):
        """Update an existing analysis session"""
        db = self.get_session()
        try:
            session = db.query(AnalysisSession).filter(AnalysisSession.session_id == session_id).first()
            if session:
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                db.commit()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error updating session: {e}")
            return False
        finally:
            db.close()
    
    def save_image_metadata(self, session_id: str, filename: str, 
                          file_size: int, width: int, height: int) -> bool:
        """Save image metadata to database"""
        db = self.get_session()
        try:
            metadata = ImageMetadata(
                session_id=session_id,
                filename=filename,
                file_size=file_size,
                image_width=width,
                image_height=height
            )
            db.add(metadata)
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error saving image metadata: {e}")
            return False
        finally:
            db.close()
    
    def save_stitching_result(self, session_id: str, result_width: int, 
                            result_height: int, processing_time: float,
                            success: bool, settings: Dict, 
                            error_message: Optional[str] = None) -> bool:
        """Save stitching result to database"""
        db = self.get_session()
        try:
            result = StitchingResult(
                session_id=session_id,
                result_width=result_width,
                result_height=result_height,
                processing_time=processing_time,
                success=success,
                error_message=error_message,
                settings=settings
            )
            db.add(result)
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error saving stitching result: {e}")
            return False
        finally:
            db.close()
    
    def save_ndvi_analysis(self, session_id: str, image_type: str,
                          analysis_data: Dict, processing_time: float,
                          band_configuration: str, 
                          image_filename: Optional[str] = None) -> bool:
        """Save NDVI analysis result to database"""
        db = self.get_session()
        try:
            analysis = NDVIAnalysis(
                session_id=session_id,
                image_type=image_type,
                image_filename=image_filename,
                band_configuration=band_configuration,
                mean_ndvi=analysis_data.get('mean_ndvi', 0.0),
                std_ndvi=analysis_data.get('std_ndvi', 0.0),
                min_ndvi=analysis_data.get('min_ndvi', 0.0),
                max_ndvi=analysis_data.get('max_ndvi', 0.0),
                vegetation_coverage=analysis_data.get('vegetation_coverage', 0.0),
                dense_vegetation_percentage=analysis_data.get('dense_vegetation_percentage', 0.0),
                moderate_vegetation_percentage=analysis_data.get('moderate_vegetation_percentage', 0.0),
                sparse_vegetation_percentage=analysis_data.get('sparse_vegetation_percentage', 0.0),
                bare_soil_percentage=analysis_data.get('bare_soil_percentage', 0.0),
                water_percentage=analysis_data.get('water_percentage', 0.0),
                processing_time=processing_time
            )
            db.add(analysis)
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error saving NDVI analysis: {e}")
            return False
        finally:
            db.close()
    
    def get_session_history(self, limit: int = 50) -> List[Dict]:
        """Get recent analysis sessions"""
        db = self.get_session()
        try:
            sessions = db.query(AnalysisSession)\
                       .order_by(AnalysisSession.created_at.desc())\
                       .limit(limit)\
                       .all()
            
            result = []
            for session in sessions:
                result.append({
                    'session_id': session.session_id,
                    'session_name': session.session_name,
                    'created_at': session.created_at,
                    'total_images': session.total_images,
                    'stitching_completed': session.stitching_completed,
                    'ndvi_enabled': session.ndvi_enabled,
                    'detector_type': session.detector_type
                })
            return result
        except SQLAlchemyError as e:
            print(f"Error getting session history: {e}")
            return []
        finally:
            db.close()
    
    def get_session_details(self, session_id: str) -> Optional[Dict]:
        """Get detailed information about a specific session"""
        db = self.get_session()
        try:
            session = db.query(AnalysisSession).filter(AnalysisSession.session_id == session_id).first()
            if not session:
                return None
            
            # Get associated images
            images = db.query(ImageMetadata).filter(ImageMetadata.session_id == session_id).all()
            
            # Get stitching results
            stitching = db.query(StitchingResult).filter(StitchingResult.session_id == session_id).first()
            
            # Get NDVI analyses
            ndvi_analyses = db.query(NDVIAnalysis).filter(NDVIAnalysis.session_id == session_id).all()
            
            return {
                'session': {
                    'session_id': session.session_id,
                    'session_name': session.session_name,
                    'created_at': session.created_at,
                    'total_images': session.total_images,
                    'stitching_completed': session.stitching_completed,
                    'ndvi_enabled': session.ndvi_enabled,
                    'detector_type': session.detector_type,
                    'settings': {
                        'match_threshold': session.match_threshold,
                        'ransac_threshold': session.ransac_threshold,
                        'band_type': session.band_type
                    }
                },
                'images': [
                    {
                        'filename': img.filename,
                        'file_size': img.file_size,
                        'dimensions': f"{img.image_width}x{img.image_height}",
                        'upload_time': img.upload_time,
                        'processed': img.processed
                    }
                    for img in images
                ],
                'stitching_result': {
                    'success': stitching.success if stitching else False,
                    'dimensions': f"{stitching.result_width}x{stitching.result_height}" if stitching else None,
                    'processing_time': stitching.processing_time if stitching else None,
                    'error_message': stitching.error_message if stitching else None
                } if stitching else None,
                'ndvi_analyses': [
                    {
                        'image_type': analysis.image_type,
                        'image_filename': analysis.image_filename,
                        'band_configuration': analysis.band_configuration,
                        'mean_ndvi': analysis.mean_ndvi,
                        'vegetation_coverage': analysis.vegetation_coverage,
                        'processing_time': analysis.processing_time,
                        'created_at': analysis.created_at
                    }
                    for analysis in ndvi_analyses
                ]
            }
        except SQLAlchemyError as e:
            print(f"Error getting session details: {e}")
            return None
        finally:
            db.close()
    
    def get_analytics_summary(self) -> Dict:
        """Get summary analytics across all sessions"""
        db = self.get_session()
        try:
            # Total sessions
            total_sessions = db.query(AnalysisSession).count()
            
            # Total images processed
            total_images = db.query(ImageMetadata).count()
            
            # Successful stitching operations
            successful_stitching = db.query(StitchingResult).filter(StitchingResult.success == True).count()
            
            # NDVI analyses performed
            total_ndvi_analyses = db.query(NDVIAnalysis).count()
            
            # Average processing times
            avg_stitching_time = db.query(StitchingResult.processing_time).filter(
                StitchingResult.success == True
            ).all()
            avg_stitching_time = sum(t[0] for t in avg_stitching_time) / len(avg_stitching_time) if avg_stitching_time else 0
            
            avg_ndvi_time = db.query(NDVIAnalysis.processing_time).all()
            avg_ndvi_time = sum(t[0] for t in avg_ndvi_time) / len(avg_ndvi_time) if avg_ndvi_time else 0
            
            # Most used detector type
            detector_usage = db.query(AnalysisSession.detector_type).all()
            detector_counts = {}
            for detector in detector_usage:
                detector_counts[detector[0]] = detector_counts.get(detector[0], 0) + 1
            most_used_detector = max(detector_counts.items(), key=lambda x: x[1])[0] if detector_counts else "N/A"
            
            return {
                'total_sessions': total_sessions,
                'total_images_processed': total_images,
                'successful_stitching_operations': successful_stitching,
                'total_ndvi_analyses': total_ndvi_analyses,
                'average_stitching_time': round(avg_stitching_time, 2),
                'average_ndvi_time': round(avg_ndvi_time, 2),
                'most_used_detector': most_used_detector,
                'detector_usage': detector_counts
            }
        except SQLAlchemyError as e:
            print(f"Error getting analytics summary: {e}")
            return {}
        finally:
            db.close()
    
    def export_session_data(self, session_id: str) -> Optional[pd.DataFrame]:
        """Export session data as pandas DataFrame for analysis"""
        session_details = self.get_session_details(session_id)
        if not session_details:
            return None
        
        # Create a comprehensive dataset
        data = []
        
        # Add session info
        session_info = session_details['session']
        base_row = {
            'session_id': session_info['session_id'],
            'session_name': session_info['session_name'],
            'created_at': session_info['created_at'],
            'detector_type': session_info['detector_type'],
            'match_threshold': session_info['settings']['match_threshold'],
            'ransac_threshold': session_info['settings']['ransac_threshold'],
            'band_type': session_info['settings']['band_type']
        }
        
        # Add NDVI analysis data
        for analysis in session_details['ndvi_analyses']:
            row = base_row.copy()
            row.update({
                'analysis_type': 'ndvi',
                'image_type': analysis['image_type'],
                'image_filename': analysis['image_filename'],
                'band_configuration': analysis['band_configuration'],
                'mean_ndvi': analysis['mean_ndvi'],
                'vegetation_coverage': analysis['vegetation_coverage'],
                'processing_time': analysis['processing_time']
            })
            data.append(row)
        
        return pd.DataFrame(data) if data else None

# Initialize database manager
db_manager = DatabaseManager()

def initialize_database():
    """Initialize the database with required tables"""
    return db_manager.create_tables()