import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

def send_alert_email(recipient="abhinavaby07@gmail.com", alert_type="General", image_path=None):
    # Hardcoded sender credentials
    sender_email = "abhinavaby07@gmail.com"
    sender_password = "ussnbljlnrjtdlow"

    subject = f"URGENT: Drowsiness System Alert - {alert_type}"
    body = f"""
    The Advanced Drowsiness Detection system has detected a prolonged critical state: {alert_type}.
    
    This alert has been active for more than 8 seconds. Please check on the driver immediately.
    """
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                image = MIMEImage(img_data, name=os.path.basename(image_path))
                msg.attach(image)
        except Exception as e:
            print(f"Failed to attach image: {e}")
    
    try:
        # Use Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient, text)
        server.quit()
        print(f"Alert email successfully sent to {recipient}")
        return True
    except Exception as e:
        print(f"Failed to send email alert: {e}")
        return False
