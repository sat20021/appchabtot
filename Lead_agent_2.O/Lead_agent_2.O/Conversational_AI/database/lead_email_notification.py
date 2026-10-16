"""
Lead Email Notification Service

Handles automatic email notifications to sales team when a lead provides their email
during the conversational form filling process.

IMPORTANT: This module is triggered by CODE LOGIC (not LLM) to avoid hallucinations.
"""
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def get_sales_recipients(mongo_db) -> List[str]:
    """
    Query lead_email_alerts collection for recipients with type='All'.
    
    Args:
        mongo_db: MongoDB database instance
        
    Returns:
        List of email addresses to notify
    """
    try:
        # Query for all recipients with type="All"
        recipients = list(mongo_db.lead_email_alerts.find({"type": "All"}))
        
        if not recipients:
            logger.warning("[lead-notification] No recipients found in lead_email_alerts with type='All'")
            return []
        
        # Extract email addresses
        recipient_emails = [r.get("email") for r in recipients if r.get("email")]
        
        logger.info(f"[lead-notification] Found {len(recipient_emails)} sales recipient(s): {recipient_emails}")
        return recipient_emails
        
    except Exception as e:
        logger.error(f"[lead-notification] Error querying lead_email_alerts collection: {e}")
        return []


def should_send_notification(lead_id: str, mongo_db) -> bool:
    """
    Check if notification has already been sent for this lead.
    
    Args:
        lead_id: Lead ID (e.g., LD-259)
        mongo_db: MongoDB database instance
        
    Returns:
        True if notification should be sent, False if already sent
    """
    try:
        # Query lead document for notification flag
        lead = mongo_db.leads.find_one(
            {"_id": lead_id},
            {"email_notification_sent": 1}
        )
        
        if not lead:
            logger.warning(f"[lead-notification] Lead {lead_id} not found in database")
            return False
        
        # Check if notification already sent
        already_sent = lead.get("email_notification_sent", False)
        
        if already_sent:
            logger.info(f"[lead-notification] Notification already sent for lead {lead_id}, skipping")
            return False
        
        logger.info(f"[lead-notification] Notification not yet sent for lead {lead_id}")
        return True
        
    except Exception as e:
        logger.error(f"[lead-notification] Error checking notification status for {lead_id}: {e}")
        return False


def mark_notification_sent(lead_id: str, mongo_db) -> None:
    """
    Update lead document to mark notification as sent.
    
    Args:
        lead_id: Lead ID (e.g., LD-259)
        mongo_db: MongoDB database instance
    """
    try:
        now_iso = datetime.utcnow().isoformat()
        
        mongo_db.leads.update_one(
            {"_id": lead_id},
            {
                "$set": {
                    "email_notification_sent": True,
                    "email_notification_sent_at": now_iso
                }
            }
        )
        
        logger.info(f"[lead-notification] Marked notification as sent for lead {lead_id}")
        
    except Exception as e:
        logger.error(f"[lead-notification] Error marking notification as sent for {lead_id}: {e}")


def send_lead_email_notification(lead_email: str, lead_id: str, mongo_db) -> bool:
    """
    Send notification to sales team when lead provides email.
    
    Email format (as per user requirements):
    "We have got a lead from the mail {email}, and the data has been stored in leads with lead id: {lead_id}."
    
    Args:
        lead_email: The lead's email address
        lead_id: The generated lead ID (e.g., LD-259)
        mongo_db: MongoDB database instance
        
    Returns:
        True if notification sent successfully, False otherwise
    """
    try:
        # Step 1: Get sales recipients
        recipients = get_sales_recipients(mongo_db)
        
        if not recipients:
            logger.warning("[lead-notification] No recipients to notify, skipping email send")
            return False
        
        # Step 2: Prepare email content
        subject = f"New Lead Alert: {lead_email}"
        
        body = f"""We have got a lead from the mail {lead_email}, and the data has been stored in leads with lead id: {lead_id}.

---
This is an automated notification from the Lead Form system.
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        
        # Step 3: Import email sender (lazy import to avoid circular dependencies)
        try:
            import sys
            from pathlib import Path
            
            # Add Conversational_AI to path to find utils module
            current_path = Path(__file__).resolve()
            conversational_ai_path = current_path.parents[1]  # Go up to Conversational_AI directory
            if str(conversational_ai_path) not in sys.path:
                sys.path.insert(0, str(conversational_ai_path))
            
            from utils.centralized_email import send_leads_email
            
        except ImportError as e:
            logger.error(f"[lead-notification] Failed to import email sender: {e}")
            return False
        
        # Step 4: Send email to all recipients
        success_count = 0
        fail_count = 0
        
        for recipient in recipients:
            try:
                result = send_leads_email(
                    to_email=recipient,
                    subject=subject,
                    body=body,
                    is_html=False
                )
                
                if result.get("success"):
                    logger.info(f"[lead-notification] ✅ Sent notification to {recipient}")
                    success_count += 1
                else:
                    logger.warning(f"[lead-notification] ⚠️ Failed to send to {recipient}: {result.get('message')}")
                    fail_count += 1
                    
            except Exception as e:
                logger.error(f"[lead-notification] ❌ Error sending to {recipient}: {e}")
                fail_count += 1
        
        # Step 5: Mark notification as sent (even if some emails failed)
        if success_count > 0:
            mark_notification_sent(lead_id, mongo_db)
            logger.info(f"[lead-notification] Notification summary: {success_count} succeeded, {fail_count} failed")
            return True
        else:
            logger.error(f"[lead-notification] All email sends failed for lead {lead_id}")
            return False
        
    except Exception as e:
        logger.error(f"[lead-notification] Unexpected error in send_lead_email_notification: {e}")
        return False
