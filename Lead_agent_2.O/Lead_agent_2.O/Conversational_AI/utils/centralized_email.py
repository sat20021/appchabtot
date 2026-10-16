"""
Centralized Email Service for Lead Notifications

This module provides a unified interface for sending lead notification emails.
Uses the main backend's email service via HTTP API or direct import when available.
"""
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def send_leads_email(to_email: str, subject: str, body: str, is_html: bool = False) -> Dict[str, Any]:
    """
    Send a lead notification email to the specified recipient.
    
    This function attempts multiple strategies to send emails:
    1. Try to use the main backend's email service (if accessible)
    2. Fallback to SMTP configuration from environment variables
    3. Log the email if no sending method is available
    
    Args:
        to_email: Recipient email address
        subject: Email subject line
        body: Email body content
        is_html: Whether the body is HTML formatted
        
    Returns:
        Dict with 'success' (bool) and 'message' (str) keys
    """
    # Strategy 1: Try to import and use main backend email service
    try:
        # Attempt to import from the main Backend app
        import sys
        from pathlib import Path
        
        # Find the Backend directory
        current_file = Path(__file__).resolve()
        backend_path = None
        for parent in current_file.parents:
            if parent.name == "Backend":
                backend_path = parent
                break
            # Also check if we're in the nested Lead_agent_2.O structure
            if (parent / "app" / "services" / "email_service.py").exists():
                backend_path = parent
                break
        
        if backend_path and str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))
        
        # Try to import the email service
        from app.services.email_service import send_email
        
        # Use the main backend's email service
        result = send_email(
            to_email=to_email,
            subject=subject,
            body=body,
            is_html=is_html
        )
        logger.info(f"[centralized_email] Sent email via main backend service to {to_email}")
        return {"success": True, "message": "Email sent via main backend service"}
        
    except ImportError:
        logger.debug("[centralized_email] Main backend email service not available, trying fallback")
    except Exception as e:
        logger.warning(f"[centralized_email] Backend email service failed: {e}")
    
    # Strategy 2: Try SMTP with environment variables
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("FROM_EMAIL", smtp_user)
        
        if not all([smtp_host, smtp_user, smtp_password]):
            raise ValueError("SMTP configuration incomplete")
        
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        content_type = 'html' if is_html else 'plain'
        msg.attach(MIMEText(body, content_type))
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info(f"[centralized_email] Sent email via SMTP to {to_email}")
        return {"success": True, "message": "Email sent via SMTP"}
        
    except Exception as e:
        logger.warning(f"[centralized_email] SMTP fallback failed: {e}")
    
    # Strategy 3: Log the email for manual processing
    logger.info(f"[centralized_email] EMAIL QUEUED (not sent) - To: {to_email}, Subject: {subject}")
    logger.info(f"[centralized_email] Body: {body[:200]}...")
    
    return {
        "success": False, 
        "message": "Email could not be sent - no email service available. Email logged for manual processing."
    }


# Convenience function alias for backward compatibility
send_lead_notification = send_leads_email
