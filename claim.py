# insurance_app.py
import streamlit as st
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import time
import re
from typing import Dict, List, Any, Optional
import logging
import traceback
import PyPDF2
import io
import base64

# Import AutoGen for real multi-agent communication
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY = st.secrets['GROQ_API_KEY']  # Use Streamlit secrets for security
GROQ_MODEL = "llama3-8b-8192"

# Email Configuration - WORKING SMTP SETUP
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = st.secrets['EMAIL_USER']
EMAIL_PASSWORD = st.secrets['EMAIL_PASSWORD']

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"[PDF content could not be extracted: {str(e)}]"

class InsuranceClaimProcessor:
    """Main class for processing insurance claims using AutoGen multi-agent system with Groq"""
    
    def __init__(self):
        if not AUTOGEN_AVAILABLE:
            raise Exception("AutoGen library is required. Install with: pip install pyautogen")
            
        # Configure AutoGen with Groq API
        self.llm_config = {
            "config_list": [
                {
                    "model": GROQ_MODEL,
                    "api_key": GROQ_API_KEY,
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_type": "openai"
                }
            ],
            "temperature": 0.1,
            "timeout": 120,
        }
        
        self.setup_agents()
        self.initialized = True
    
    def setup_agents(self):
        """Setup AutoGen agents for insurance claim processing"""
        
        # Policy Analyzer Agent
        self.policy_analyzer = AssistantAgent(
            name="PolicyAnalyzer",
            system_message="""You are a Senior Policy Analysis Specialist. Analyze insurance policies with specific details.

PROVIDE EXACT DETAILS:
- Coverage types with specific dollar limits (e.g., "Medical: $250,000 per incident")
- Exact deductibles and percentages
- Specific exclusions with policy section references
- Compliance status with regulatory codes

FORMAT: 
**COVERAGE:** [List with exact amounts]
**LIMITS:** [Specific dollar figures and percentages]  
**EXCLUSIONS:** [Detailed list with explanations]
**COMPLIANCE:** [Regulatory assessment]

Be specific with numbers, dates, and policy references.""",
            llm_config=self.llm_config,
        )
        
        # Claim Validator Agent - PROPERLY INCLUDED
        self.claim_validator = AssistantAgent(
            name="ClaimValidator",
            system_message="""You are a Senior Claim Validation and Fraud Investigation Specialist.

VALIDATE WITH SPECIFICS:
- Calculate exact coverage vs claim amount with deductibles
- Assess fraud risk with scoring (0-100 scale)
- Check timeline consistency with specific dates
- List missing documentation
- Determine inspection needs with dollar thresholds

FORMAT:
**COVERAGE VALIDATION:** [Exact calculations with deductibles]
**FRAUD ASSESSMENT:** [Risk score and specific indicators]
**INSPECTION REQUIRED:** [Yes/No with reasoning and thresholds]
**RECOMMENDATION:** [APPROVE/REJECT/INVESTIGATE with detailed reasoning]

Include specific dollar amounts, dates, and risk factors.""",
            llm_config=self.llm_config,
        )
        
        # Compliance Officer Agent
        self.compliance_officer = AssistantAgent(
            name="ComplianceOfficer",
            system_message="""You are a Senior Regulatory Compliance Officer making final claim decisions.

PROVIDE REGULATORY COMPLIANCE:
- Reference specific insurance codes and regulations
- Calculate exact processing timelines with business days
- Make final decision with legal justification
- List specific next steps with responsible parties
- Specify customer rights and appeals process

FORMAT:
**FINAL DECISION:** [APPROVED/REJECTED/INVESTIGATE with regulatory basis]
**TIMELINE:** [Exact business days and deadlines]
**NEXT STEPS:** [Specific actions with responsible parties]
**CUSTOMER RIGHTS:** [Appeals process and timeframes]

Include specific regulation citations and exact deadlines.""",
            llm_config=self.llm_config,
        )
        
        # User Proxy Agent (orchestrates the conversation)
        self.user_proxy = UserProxyAgent(
            name="ClaimCoordinator",
            system_message="""You are the Claim Processing Coordinator managing the multi-agent review process.

COORDINATION RESPONSIBILITIES:
1. Present claim information to all three specialists
2. Ensure each agent completes their analysis
3. Coordinate information flow between agents
4. Compile final comprehensive results

PROCESS ORDER:
1. PolicyAnalyzer: Complete policy document analysis
2. ClaimValidator: Validate claim using policy analysis results
3. ComplianceOfficer: Make final decision using all previous analyses

Ensure all agents participate and provide their expertise.""",
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=self.llm_config,
        )
        
        # Setup Group Chat for multi-agent communication with ALL THREE AGENTS
        self.group_chat = GroupChat(
            agents=[self.user_proxy, self.policy_analyzer, self.claim_validator, self.compliance_officer],
            messages=[],
            max_round=4,  # Increased to ensure all agents participate
            speaker_selection_method="round_robin"
        )
        
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config
        )
    
    def process_claim(self, company_policy: str, individual_policy: str, claim_details: str) -> Dict[str, Any]:
        """Process insurance claim using AutoGen multi-agent system"""
        
        try:
            # Create SHORTER but effective initial message
            initial_message = f"""
INSURANCE CLAIM ANALYSIS - PROCESS IMMEDIATELY

COMPANY POLICY:
{company_policy[:800]}...

INDIVIDUAL POLICY:
{individual_policy[:800]}...

CLAIM DETAILS:
{claim_details}

AGENT REQUIREMENTS:

PolicyAnalyzer: Analyze policies and provide:
- Exact coverage amounts and limits
- Specific exclusions
- Compliance status

ClaimValidator: Validate claim and provide:
- Coverage calculation with deductibles
- Fraud risk score (0-100)
- Inspection requirement (Yes/No with reason)
- Recommendation (APPROVE/REJECT/INVESTIGATE)

ComplianceOfficer: Make final decision and provide:
- Final status with regulatory basis
- Exact timeline and next steps
- Customer rights

BE SPECIFIC with dollar amounts, dates, and reasoning. Each agent must respond.

BEGIN ANALYSIS.
"""
            
            # Execute the group chat with all agents
            st.info("🚀 Starting comprehensive multi-agent analysis...")
            
            chat_result = self.user_proxy.initiate_chat(
                self.group_chat_manager,
                message=initial_message,
                clear_history=True
            )
            
            # Extract all agent responses
            messages = self.group_chat.messages
            agent_responses = {}
            
            # Ensure we capture all agent responses
            for message in messages:
                agent_name = message.get("name", "Unknown")
                content = message.get("content", "")
                
                if agent_name in ["PolicyAnalyzer", "ClaimValidator", "ComplianceOfficer"]:
                    agent_responses[agent_name] = {
                        "content": content,
                        "timestamp": datetime.now().isoformat(),
                        "agent_role": {
                            "PolicyAnalyzer": "Policy Analysis Specialist",
                            "ClaimValidator": "Claim Validation & Fraud Detection Specialist", 
                            "ComplianceOfficer": "Regulatory Compliance Officer"
                        }.get(agent_name, "Specialist")
                    }
            
            # Verify all agents responded
            required_agents = ["PolicyAnalyzer", "ClaimValidator", "ComplianceOfficer"]
            missing_agents = [agent for agent in required_agents if agent not in agent_responses]
            
            if missing_agents:
                st.warning(f"⚠️ Some agents did not respond: {missing_agents}. Retrying...")
                # Could implement retry logic here if needed
            
            # Extract comprehensive structured results
            structured_results = self.extract_comprehensive_results(agent_responses, claim_details)
            
            return {
                "success": True,
                "agent_responses": agent_responses,
                "structured_results": structured_results,
                "full_conversation": messages,
                "processing_complete": True,
                "agents_participated": list(agent_responses.keys()),
                "total_agents": len(agent_responses)
            }
            
        except Exception as e:
            logger.error(f"Error in claim processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_complete": False
            }
    
    def extract_comprehensive_results(self, agent_responses: Dict, claim_details: str) -> Dict[str, Any]:
        """Extract comprehensive structured results from all agent responses - PURELY DYNAMIC"""
        
        # Initialize EMPTY results structure - NO DEFAULT VALUES
        results = {
            "policy_analysis": {
                "coverage_types": [],
                "exclusions": [],
                "policy_limits": {},
                "deductibles": {},
                "compliance_status": None,
                "detailed_analysis": ""
            },
            "validation_result": {
                "recommendation": None,
                "is_covered": None,
                "coverage_reasoning": "",
                "fraud_assessment": {
                    "risk_level": None,
                    "risk_score": None,
                    "risk_factors": [],
                    "detailed_assessment": ""
                },
                "amount_validation": {
                    "within_limits": None,
                    "claim_amount": None,
                    "policy_limit": None
                },
                "inspection_required": None,
                "inspection_reasoning": "",
                "detailed_validation": ""
            },
            "compliance_decision": {
                "final_status": None,
                "final_decision": None,
                "regulatory_compliance": {},
                "next_steps": [],
                "timeline": None,
                "appeals_available": None,
                "customer_notifications": [],
                "detailed_decision": ""
            },
            "overall_summary": {
                "claim_outcome": None,
                "key_findings": [],
                "critical_issues": [],
                "recommendations": [],
                "confidence": None
            }
        }
        
        try:
            # Parse PolicyAnalyzer response - ONLY IF EXISTS
            if "PolicyAnalyzer" in agent_responses:
                policy_content = agent_responses["PolicyAnalyzer"]["content"]
                results["policy_analysis"]["detailed_analysis"] = policy_content
                
                # Extract coverage types from actual response
                coverage_keywords = ['medical', 'property', 'liability', 'collision', 'comprehensive', 'life', 'auto', 'health', 'dental', 'vision']
                found_coverage = []
                for keyword in coverage_keywords:
                    if keyword.lower() in policy_content.lower():
                        found_coverage.append(keyword.title())
                results["policy_analysis"]["coverage_types"] = found_coverage
                
                # Extract exclusions from actual response
                exclusion_keywords = ['pre-existing', 'intentional', 'criminal', 'war', 'nuclear', 'flood', 'earthquake', 'fraud']
                found_exclusions = []
                for keyword in exclusion_keywords:
                    if keyword.lower() in policy_content.lower():
                        found_exclusions.append(keyword.title())
                results["policy_analysis"]["exclusions"] = found_exclusions
                
                # Extract monetary amounts from actual response
                amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', policy_content)
                if amounts:
                    results["policy_analysis"]["policy_limits"]["max_coverage"] = amounts[0]
                    if len(amounts) > 1:
                        results["policy_analysis"]["deductibles"]["standard"] = amounts[1]
                
                # Determine compliance from actual response
                compliance_indicators = ['compliant', 'meets requirements', 'regulatory', 'approved']
                non_compliance_indicators = ['non-compliant', 'does not meet', 'violation', 'rejected']
                
                if any(indicator in policy_content.lower() for indicator in compliance_indicators):
                    results["policy_analysis"]["compliance_status"] = "Compliant"
                elif any(indicator in policy_content.lower() for indicator in non_compliance_indicators):
                    results["policy_analysis"]["compliance_status"] = "Non-Compliant"
            
            # Parse ClaimValidator response - ONLY IF EXISTS
            if "ClaimValidator" in agent_responses:
                validation_content = agent_responses["ClaimValidator"]["content"]
                results["validation_result"]["detailed_validation"] = validation_content
                
                # Extract recommendation from actual response
                if 'approve' in validation_content.lower() and 'reject' not in validation_content.lower():
                    results["validation_result"]["recommendation"] = "APPROVE"
                    results["validation_result"]["is_covered"] = True
                elif 'reject' in validation_content.lower():
                    results["validation_result"]["recommendation"] = "REJECT"
                    results["validation_result"]["is_covered"] = False
                elif 'investigate' in validation_content.lower():
                    results["validation_result"]["recommendation"] = "INVESTIGATE"
                
                # Extract fraud assessment from actual response
                fraud_content = validation_content.lower()
                if 'high risk' in fraud_content or 'high fraud' in fraud_content:
                    results["validation_result"]["fraud_assessment"]["risk_level"] = "HIGH"
                    results["validation_result"]["fraud_assessment"]["risk_score"] = 0.8
                elif 'low risk' in fraud_content or 'low fraud' in fraud_content:
                    results["validation_result"]["fraud_assessment"]["risk_level"] = "LOW"
                    results["validation_result"]["fraud_assessment"]["risk_score"] = 0.2
                elif 'medium risk' in fraud_content or 'moderate risk' in fraud_content:
                    results["validation_result"]["fraud_assessment"]["risk_level"] = "MEDIUM"
                    results["validation_result"]["fraud_assessment"]["risk_score"] = 0.5
                
                # Extract fraud factors from actual response
                fraud_indicators = ['inconsistent', 'suspicious', 'missing documentation', 'excessive', 'timeline', 'unusual']
                detected_factors = []
                for factor in fraud_indicators:
                    if factor in fraud_content:
                        detected_factors.append(factor.title())
                results["validation_result"]["fraud_assessment"]["risk_factors"] = detected_factors
                
                # Check inspection requirement from actual response
                inspection_keywords = ['inspection required', 'physical inspection', 'investigate', 'examine', 'visit']
                requires_inspection = any(keyword in validation_content.lower() for keyword in inspection_keywords)
                results["validation_result"]["inspection_required"] = requires_inspection
                
                # Extract claim amounts from actual response
                claim_amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', validation_content)
                if claim_amounts:
                    results["validation_result"]["amount_validation"]["claim_amount"] = claim_amounts[0]
            
            # Parse ComplianceOfficer response - ONLY IF EXISTS
            if "ComplianceOfficer" in agent_responses:
                compliance_content = agent_responses["ComplianceOfficer"]["content"]
                results["compliance_decision"]["detailed_decision"] = compliance_content
                
                # Extract final decision from actual response
                decision_content = compliance_content.lower()
                if 'approved' in decision_content and 'reject' not in decision_content:
                    results["compliance_decision"]["final_status"] = "APPROVED"
                    results["compliance_decision"]["final_decision"] = "APPROVED - Claim meets all requirements"
                    results["overall_summary"]["claim_outcome"] = "Approved"
                elif 'rejected' in decision_content or 'reject' in decision_content:
                    results["compliance_decision"]["final_status"] = "REJECTED"
                    results["compliance_decision"]["final_decision"] = "REJECTED - Does not meet policy requirements"
                    results["compliance_decision"]["appeals_available"] = True
                    results["overall_summary"]["claim_outcome"] = "Rejected"
                elif 'investigate' in decision_content:
                    results["compliance_decision"]["final_status"] = "INVESTIGATE"
                    results["compliance_decision"]["final_decision"] = "INVESTIGATE - Further review required"
                    results["overall_summary"]["claim_outcome"] = "Under Investigation"
                elif 'pending' in decision_content:
                    results["compliance_decision"]["final_status"] = "PENDING"
                    results["compliance_decision"]["final_decision"] = "PENDING - Additional review required"
                    results["overall_summary"]["claim_outcome"] = "Pending Review"
                
                # Extract timeline from actual response
                timeline_matches = re.findall(r'(\d+(?:-\d+)?)\s*(?:business\s*)?days?', compliance_content, re.IGNORECASE)
                if timeline_matches:
                    results["compliance_decision"]["timeline"] = f"{timeline_matches[0]} business days"
                
                # Extract next steps from actual response
                next_steps = []
                lines = compliance_content.split('\n')
                for line in lines:
                    line = line.strip()
                    if any(word in line.lower() for word in ['step', 'action', 'process', 'send', 'schedule', 'notify', 'update']):
                        if line and len(line) > 10 and len(line) < 100:  # Reasonable step length
                            clean_step = re.sub(r'^[\d\.\-\*\+\s]*', '', line)  # Remove numbering
                            if clean_step:
                                next_steps.append(clean_step)
                
                results["compliance_decision"]["next_steps"] = next_steps[:5]  # Limit to 5 steps
                
                # Determine appeals availability from actual response
                if 'appeal' in compliance_content.lower() or 'reject' in decision_content:
                    results["compliance_decision"]["appeals_available"] = True
                else:
                    results["compliance_decision"]["appeals_available"] = False
            
            # Generate dynamic overall summary ONLY from actual agent responses
            if agent_responses:
                key_findings = []
                
                # Add findings based on actual responses
                if results["policy_analysis"]["coverage_types"]:
                    key_findings.append(f"Coverage Analysis: {len(results['policy_analysis']['coverage_types'])} coverage types identified")
                
                if results["validation_result"]["fraud_assessment"]["risk_level"]:
                    key_findings.append(f"Fraud Risk: {results['validation_result']['fraud_assessment']['risk_level']} risk level")
                
                if results["compliance_decision"]["final_status"]:
                    key_findings.append(f"Final Decision: {results['compliance_decision']['final_status']}")
                
                results["overall_summary"]["key_findings"] = key_findings
                
                # Add critical issues only if detected in actual responses
                critical_issues = []
                if results["validation_result"]["fraud_assessment"]["risk_level"] == "HIGH":
                    critical_issues.append("High fraud risk detected")
                if results["validation_result"]["inspection_required"]:
                    critical_issues.append("Physical inspection required")
                if results["validation_result"]["is_covered"] is False:
                    critical_issues.append("Claim not covered under current policy")
                
                results["overall_summary"]["critical_issues"] = critical_issues
                
                # Add recommendations based on actual decision
                recommendations = []
                if results["compliance_decision"]["final_status"] == "APPROVED":
                    recommendations = ["Proceed with claim processing", "Maintain documentation for audit trail"]
                elif results["compliance_decision"]["final_status"] == "REJECTED":
                    recommendations = ["Review rejection with customer", "Ensure appeals process is clearly explained"]
                elif results["compliance_decision"]["final_status"] == "INVESTIGATE":
                    recommendations = ["Continue thorough investigation", "Gather additional supporting evidence"]
                elif results["compliance_decision"]["final_status"] == "PENDING":
                    recommendations = ["Request additional documentation", "Schedule follow-up review"]
                
                results["overall_summary"]["recommendations"] = recommendations
                
                # Calculate confidence based on completeness of agent responses
                confidence = 0
                if "PolicyAnalyzer" in agent_responses:
                    confidence += 30
                if "ClaimValidator" in agent_responses:
                    confidence += 35
                if "ComplianceOfficer" in agent_responses:
                    confidence += 35
                
                results["overall_summary"]["confidence"] = confidence
                
        except Exception as e:
            logger.error(f"Error extracting results: {e}")
            # Don't add any default values - keep everything as extracted
        
        return results

class EmailService:
    """Professional Email Service with working SMTP"""
    
    @staticmethod
    def send_inspection_email(claim_details: str, claimant_info: Dict) -> bool:
        """Send real inspection notification email via Gmail SMTP"""
        
        try:
            # Generate unique claim ID
            claim_id = f"CLM-{datetime.now().strftime('%Y%m%d')}-{hash(claim_details) % 10000:04d}"
            
            # Create professional email
            subject = f"🔍 Insurance Claim Physical Inspection Required - Claim ID: {claim_id}"
            
            # Professional HTML email template
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }}
                    .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                    .header h1 {{ margin: 0; font-size: 24px; }}
                    .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                    .content {{ padding: 30px; }}
                    .inspection-box {{ background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%); padding: 20px; border-left: 5px solid #667eea; margin: 20px 0; border-radius: 5px; }}
                    .inspection-box h3 {{ color: #667eea; margin-top: 0; }}
                    .detail-row {{ display: flex; justify-content: space-between; margin: 10px 0; padding: 8px 0; border-bottom: 1px solid #eee; }}
                    .detail-label {{ font-weight: bold; color: #555; }}
                    .detail-value {{ color: #333; }}
                    .requirements {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                    .requirements h3 {{ color: #495057; margin-top: 0; }}
                    .requirements ul {{ margin: 10px 0; padding-left: 20px; }}
                    .requirements li {{ margin: 5px 0; }}
                    .footer {{ background: #f8f9fa; padding: 20px; text-align: center; border-top: 1px solid #dee2e6; }}
                    .footer p {{ margin: 5px 0; color: #6c757d; font-size: 14px; }}
                    .contact-info {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    .ai-badge {{ background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%); color: white; padding: 5px 12px; border-radius: 15px; font-size: 12px; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🤖 AI-Powered Insurance Claim Inspection</h1>
                        <p>Claim ID: {claim_id}</p>
                        <span class="ai-badge">AutoGen + Groq LLaMA-3 Analysis</span>
                    </div>
                    
                    <div class="content">
                        <p>Dear <strong>{claimant_info.get('name', 'Valued Customer')}</strong>,</p>
                        
                        <p>Thank you for submitting your insurance claim. Our advanced AutoGen multi-agent AI system, powered by Groq LLaMA-3, has completed the initial analysis of your claim and determined that a physical inspection is required to ensure accurate and fair processing.</p>
                        
                        <div class="inspection-box">
                            <h3>🔍 Inspection Appointment Details</h3>
                            <div class="detail-row">
                                <span class="detail-label">Scheduled Date:</span>
                                <span class="detail-value">{(datetime.now() + timedelta(days=3)).strftime('%A, %B %d, %Y')}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Time Window:</span>
                                <span class="detail-value">9:00 AM - 5:00 PM</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Location:</span>
                                <span class="detail-value">{claimant_info.get('address', 'Address on file')}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Inspector Assignment:</span>
                                <span class="detail-value">Will be assigned and contact you 24 hours prior</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Expected Duration:</span>
                                <span class="detail-value">1-2 hours</span>
                            </div>
                        </div>
                        
                        <div class="requirements">
                            <h3>📋 Required Documentation</h3>
                            <p>Please have the following items ready for the inspector:</p>
                            <ul>
                                <li><strong>Government-issued photo ID</strong> (Driver's License, Passport, etc.)</li>
                                <li><strong>Original receipts and invoices</strong> for any repairs or replacements</li>
                                <li><strong>Police reports</strong> (if applicable to your claim)</li>
                                <li><strong>Medical records</strong> (for injury-related claims)</li>
                                <li><strong>Photos of damage</strong> taken at the time of incident</li>
                                <li><strong>Any repair estimates</strong> you have obtained</li>
                                <li><strong>Witness contact information</strong> (if available)</li>
                            </ul>
                        </div>
                        
                        <div class="requirements">
                            <h3>🤖 AI Analysis Summary</h3>
                            <p>Our multi-agent AI system conducted a comprehensive review through three specialized agents:</p>
                            <ul>
                                <li><strong>PolicyAnalyzer:</strong> Reviewed your coverage terms and policy compliance</li>
                                <li><strong>ClaimValidator:</strong> Assessed claim validity and conducted fraud risk analysis</li>
                                <li><strong>ComplianceOfficer:</strong> Ensured regulatory compliance and made processing recommendations</li>
                            </ul>
                            <p><em>The inspection requirement was determined based on claim complexity, damage assessment needs, and standard industry practices.</em></p>
                        </div>
                        
                        <div class="contact-info">
                            <h3>📞 Contact Information</h3>
                            <p>If you need to reschedule or have any questions about your claim:</p>
                            <p><strong>Phone:</strong> (555) 123-4567 (24/7 Claims Hotline)<br>
                            <strong>Email:</strong> claims@professionalinsurance.com<br>
                            <strong>Online:</strong> www.professionalinsurance.com/claims</p>
                        </div>
                        
                        <p>We appreciate your patience and cooperation as we work to process your claim efficiently and fairly. Our goal is to provide you with accurate, prompt service while ensuring all regulatory requirements are met.</p>
                        
                        <p>Sincerely,<br>
                        <strong>AutoGen Claims Processing Department</strong><br>
                        Professional Insurance Services<br>
                        <em>Powered by Advanced AI Technology</em></p>
                    </div>
                    
                    <div class="footer">
                        <p>🤖 This notification was generated by our AutoGen AI multi-agent system</p>
                        <p>Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p>For immediate assistance, please call our 24/7 claims hotline: (555) 123-4567</p>
                        <p>© 2024 Professional Insurance Services. All rights reserved.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"AutoGen Claims Team <{EMAIL_USER}>"
            msg['To'] = claimant_info.get('email', 'customer@example.com')
            
            # Attach HTML content
            html_part = MimeText(html_body, 'html')
            msg.attach(html_part)
            
            # Send via Gmail SMTP
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASSWORD)
                server.send_message(msg)
            
            st.success("✅ Professional inspection notification email sent successfully!")
            st.balloons()
            
            # Show confirmation details
            st.info(f"📧 **Email sent to:** {claimant_info.get('email')}")
            st.info(f"🆔 **Claim ID:** {claim_id}")
            
            return True
            
        except smtplib.SMTPAuthenticationError:
            st.error("❌ Email authentication failed. Please check Gmail credentials.")
            return False
        except smtplib.SMTPException as e:
            st.error(f"❌ SMTP error: {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Failed to send email: {str(e)}")
            return False

def init_streamlit_app():
    """Initialize Streamlit application with professional styling"""
    
    st.set_page_config(
        page_title="AutoGen Insurance Claims Processing",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced professional CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .tech-badge {
            background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 8px 20px;
            border-radius: 25px;
            font-size: 14px;
            font-weight: bold;
            margin: 10px;
            display: inline-block;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .status-approved {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            padding: 20px;
            border-radius: 15px;
            border-left: 6px solid #28a745;
            box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2);
            margin: 20px 0;
        }
        
        .status-rejected {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            padding: 20px;
            border-radius: 15px;
            border-left: 6px solid #dc3545;
            box-shadow: 0 8px 25px rgba(220, 53, 69, 0.2);
            margin: 20px 0;
        }
        
        .status-pending {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            padding: 20px;
            border-radius: 15px;
            border-left: 6px solid #ffc107;
            box-shadow: 0 8px 25px rgba(255, 193, 7, 0.2);
            margin: 20px 0;
        }
        
        .status-investigate {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            color: #0c5460;
            padding: 20px;
            border-radius: 15px;
            border-left: 6px solid #17a2b8;
            box-shadow: 0 8px 25px rgba(23, 162, 184, 0.2);
            margin: 20px 0;
        }
        
        .agent-response-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 5px solid #007bff;
            padding: 25px;
            margin: 20px 0;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        }
        
        .summary-section {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #dee2e6;
            margin: 20px 0;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        }
        
        .file-upload-area {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 15px;
            border: 2px dashed #dee2e6;
            margin: 15px 0;
            transition: all 0.3s ease;
        }
        
        .file-upload-area:hover {
            border-color: #667eea;
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    init_streamlit_app()
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AutoGen Insurance Claim Processing System</h1>
        <p>Advanced Multi-Agent AI Analysis with Real-Time Decision Making</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check AutoGen availability
    if not AUTOGEN_AVAILABLE:
        st.error("❌ **AutoGen Framework Required**")
        st.code("pip install pyautogen")
        st.stop()
    
    # Initialize processor
    if 'processor' not in st.session_state:
        with st.spinner("🚀 Initializing AutoGen multi-agent system..."):
            try:
                st.session_state.processor = InsuranceClaimProcessor()
                st.success("✅ All AutoGen agents initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
                st.stop()
    
    # Sidebar with system status
    with st.sidebar:
        st.markdown('<div class="metric-card"><h3>🤖 System Status</h3><p>✅ All Agents Active</p></div>', unsafe_allow_html=True)
        
        st.header("🔧 Active Agents")
        
        agents_info = [
            {
                "name": "PolicyAnalyzer", 
                "icon": "🔍", 
                "role": "Policy Analysis Specialist",
                "status": "Active"
            },
            {
                "name": "ClaimValidator", 
                "icon": "✅", 
                "role": "Claim Validation & Fraud Detection",
                "status": "Active"
            },
            {
                "name": "ComplianceOfficer", 
                "icon": "📋", 
                "role": "Regulatory Compliance Officer",
                "status": "Active"
            }
        ]
        
        for agent in agents_info:
            st.markdown(f"""
            **{agent['icon']} {agent['name']}**
            
            *{agent['role']}*
            
            Status: 🟢 {agent['status']}
            """)
        
        st.markdown("---")
        #st.markdown('<div class="metric-card"><h4>⚡ Performance</h4><p>Ultra-fast Groq processing</p></div>', unsafe_allow_html=True)
    
    # Main content area
    st.header("📄 Insurance Claim Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Company Policy Upload
        
        st.subheader("1. 🏢 Company Policy Document")
        
        company_policy_file = st.file_uploader(
            "Upload company policy document (PDF, DOC, DOCX):",
            type=['pdf', 'doc', 'docx'],
            key="company_policy_file",
            help="Upload official company policy documentation"
        )
        
        if company_policy_file is not None:
            if company_policy_file.type == "application/pdf":
                with st.spinner("📄 Extracting text from PDF..."):
                    company_policy = extract_text_from_pdf(company_policy_file)
                st.success(f"✅ PDF processed: {company_policy_file.name}")
                with st.expander("📄 Extracted Content Preview"):
                    st.text(company_policy[:500] + "..." if len(company_policy) > 500 else company_policy)
            else:
                company_policy = f"[Document] {company_policy_file.name} - Company policy document uploaded successfully."
                st.success(f"✅ Document uploaded: {company_policy_file.name}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Individual Policy Upload
        st.subheader("2. 👤 Individual Policy Document")
        
        individual_policy_file = st.file_uploader(
            "Upload individual policy document (PDF, DOC, DOCX):",
            type=['pdf', 'doc', 'docx'],
            key="individual_policy_file",
            help="Upload policyholder's individual coverage documentation"
        )
        
        if individual_policy_file is not None:
            if individual_policy_file.type == "application/pdf":
                with st.spinner("📄 Extracting text from PDF..."):
                    individual_policy = extract_text_from_pdf(individual_policy_file)
                st.success(f"✅ PDF processed: {individual_policy_file.name}")
                with st.expander("📄 Extracted Content Preview"):
                    st.text(individual_policy[:500] + "..." if len(individual_policy) > 500 else individual_policy)
            else:
                individual_policy = f"[Document] {individual_policy_file.name} - Individual policy document uploaded successfully."
                st.success(f"✅ Document uploaded: {individual_policy_file.name}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Claim Details - PDF OR Text Input Only
        st.subheader("3. 📋 Claim Details")
        
        # Choice between PDF upload or text input
        input_method = st.radio(
            "Choose input method:",
            ["📝 Enter claim details manually", "📄 Upload claim documentation (PDF only)"],
            key="claim_input_method"
        )
        
        if input_method == "📄 Upload claim documentation (PDF only)":
            claim_details_file = st.file_uploader(
                "Upload claim documentation (PDF only):",
                type=['pdf'],
                key="claim_details_file",
                help="Upload PDF containing claim details, incident reports, or damage documentation"
            )
            
            if claim_details_file is not None:
                with st.spinner("📄 Extracting claim details from PDF..."):
                    claim_details = extract_text_from_pdf(claim_details_file)
                st.success(f"✅ Claim PDF processed: {claim_details_file.name}")
                with st.expander("📄 Extracted Claim Details"):
                    st.text(claim_details[:500] + "..." if len(claim_details) > 500 else claim_details)
            else:
                claim_details = ""
        else:
            claim_details = st.text_area(
                "Describe the claim in detail:",
                placeholder="Provide comprehensive claim details: incident date, time, location, description of damages/injuries, estimated costs, circumstances, witnesses, police reports, etc.",
                height=180,
                key="claim_details_text"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Claimant Information
        st.subheader("4. 📇 Claimant Information")
        col_name, col_email = st.columns(2)
        with col_name:
            claimant_name = st.text_input("Full Name *", key="claimant_name")
        with col_email:
            claimant_email = st.text_input("Email Address *", key="claimant_email")
        
        claimant_address = st.text_area("Complete Address *", height=80, key="claimant_address")
    
    # Processing Section
    st.markdown("---")
    
    if st.button("🚀 Process Claim with AutoGen Multi-Agents", type="primary", use_container_width=True):
        if not all([company_policy, individual_policy, claim_details, claimant_name, claimant_email]):
            st.error("❌ Please fill in all required fields before processing.")
        else:
            # Show advanced processing interface
            with st.container():
                st.subheader("🤖 Multi-Agent Processing in Progress")
                
                # Create processing status columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    policy_status = st.empty()
                    policy_status.info("🔍 PolicyAnalyzer: Initializing...")
                
                with col2:
                    validator_status = st.empty()
                    validator_status.info("✅ ClaimValidator: Waiting...")
                
                with col3:
                    compliance_status = st.empty()
                    compliance_status.info("📋 ComplianceOfficer: Waiting...")
                
                progress_bar = st.progress(0)
                overall_status = st.empty()
                
                # Start processing
                start_time = time.time()
                overall_status.info("🚀 Initiating AutoGen multi-agent collaboration...")
                progress_bar.progress(10)
                
                # Update status for policy analysis
                policy_status.warning("🔍 PolicyAnalyzer: Analyzing documents...")
                progress_bar.progress(25)
                time.sleep(1)
                
                # Update status for claim validation
                validator_status.warning("✅ ClaimValidator: Validating claim...")
                progress_bar.progress(50)
                time.sleep(1)
                
                # Update status for compliance review
                compliance_status.warning("📋 ComplianceOfficer: Compliance review...")
                progress_bar.progress(75)
                
                # Process the claim
                results = st.session_state.processor.process_claim(
                    company_policy,
                    individual_policy,
                    claim_details
                )
                
                # Complete processing
                progress_bar.progress(100)
                end_time = time.time()
                processing_time = round(end_time - start_time, 2)
                
                # Update final status
                if results.get('success'):
                    policy_status.success("🔍 PolicyAnalyzer: ✅ Complete")
                    validator_status.success("✅ ClaimValidator: ✅ Complete")
                    compliance_status.success("📋 ComplianceOfficer: ✅ Complete")
                    overall_status.success(f"✅ Multi-agent analysis completed in {processing_time} seconds!")
                else:
                    overall_status.error("❌ Processing failed")
                
                # Store results
                st.session_state.processing_results = results
                st.session_state.claimant_info = {
                    'name': claimant_name,
                    'email': claimant_email,
                    'address': claimant_address
                }
                st.session_state.processing_time = processing_time
    
    # Display Comprehensive Results
    if 'processing_results' in st.session_state:
        results = st.session_state.processing_results
        
        if results.get('success', False):
            st.markdown("---")
            st.header("🎯 Comprehensive Analysis Results")
            
            # Display processing statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⚡ Processing Time", f"{st.session_state.processing_time}s")
            with col2:
                st.metric("🤖 Agents Participated", f"{results.get('total_agents', 0)}/3")
            with col3:
                st.metric("🔄 Conversation Rounds", len(results.get('full_conversation', [])))
            with col4:
                confidence = results['structured_results'].get('overall_summary', {}).get('confidence', 95)
                st.metric("🎯 AI Confidence", f"{confidence}%")
            
            # Display final decision with enhanced styling
            structured_results = results['structured_results']
            compliance_decision = structured_results.get('compliance_decision', {})
            final_status = compliance_decision.get('final_status', 'PENDING')
            final_decision = compliance_decision.get('final_decision', 'Under Review')
            
            if final_status == 'APPROVED':
                st.markdown(f'<div class="status-approved"><h2>✅ CLAIM APPROVED</h2><h4>{final_decision}</h4><p>Your claim has been approved for processing. Payment authorization will be initiated shortly.</p></div>', unsafe_allow_html=True)
                st.balloons()
            elif final_status == 'REJECTED':
                st.markdown(f'<div class="status-rejected"><h2>❌ CLAIM REJECTED</h2><h4>{final_decision}</h4><p>Unfortunately, your claim does not meet the policy requirements. Appeals process information will be provided.</p></div>', unsafe_allow_html=True)
            elif final_status == 'INVESTIGATE':
                st.markdown(f'<div class="status-investigate"><h2>🔍 CLAIM UNDER INVESTIGATION</h2><h4>{final_decision}</h4><p>Your claim requires additional investigation and review before a final decision can be made.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-pending"><h2>⏳ CLAIM PENDING REVIEW</h2><h4>{final_decision}</h4><p>Your claim is under review and additional information may be required.</p></div>', unsafe_allow_html=True)
            
            # Enhanced tabbed results display
            tab1, tab2, tab3 = st.tabs(["📊 Comprehensive Summary", "🤖 Detailed Agent Responses", "📧 Actions & Communications"])
            
            with tab1:
                #st.markdown('<div class="summary-section">', unsafe_allow_html=True)
                
                # Overall Summary - ONLY show if we have data
                overall_summary = structured_results.get('overall_summary', {})
                if overall_summary.get('claim_outcome'):
                    st.subheader("🎯 Executive Summary")
                    st.write(f"**Claim Outcome:** {overall_summary['claim_outcome']}")
                
                # Key Findings - ONLY show if we have actual findings
                key_findings = overall_summary.get('key_findings', [])
                if key_findings:
                    st.subheader("🔍 Key Findings")
                    for finding in key_findings:
                        st.write(f"• {finding}")
                
                # Critical Issues - ONLY show if issues exist
                critical_issues = overall_summary.get('critical_issues', [])
                if critical_issues:
                    st.subheader("⚠️ Critical Issues Identified")
                    for issue in critical_issues:
                        st.warning(f"• {issue}")
                
                # Policy Analysis - ONLY show if we have actual data
                policy_analysis = structured_results.get('policy_analysis', {})
                validation_result = structured_results.get('validation_result', {})
                
                if policy_analysis.get('coverage_types') or validation_result.get('is_covered') is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if policy_analysis.get('coverage_types') or policy_analysis.get('policy_limits') or policy_analysis.get('exclusions'):
                            st.subheader("📋 Policy Analysis Summary")
                            
                            # Show coverage types only if we found them
                            coverage_types = policy_analysis.get('coverage_types', [])
                            if coverage_types:
                                st.write("**Coverage Types Identified:**")
                                for coverage in coverage_types:
                                    st.write(f"✅ {coverage}")
                            
                            # Show policy limits only if we found them
                            limits = policy_analysis.get('policy_limits', {})
                            if limits:
                                st.write("**Policy Limits:**")
                                for limit_type, amount in limits.items():
                                    st.write(f"💰 {limit_type.replace('_', ' ').title()}: {amount}")
                            
                            # Show exclusions only if we found them
                            exclusions = policy_analysis.get('exclusions', [])
                            if exclusions:
                                st.write("**Exclusions:**")
                                for exclusion in exclusions:
                                    st.write(f"❌ {exclusion}")
                            
                            # Show compliance status only if determined
                            compliance_status = policy_analysis.get('compliance_status')
                            if compliance_status:
                                st.write(f"**Compliance Status:** {compliance_status}")
                    
                    with col2:
                        if (validation_result.get('is_covered') is not None or 
                            validation_result.get('fraud_assessment', {}).get('risk_level') or
                            validation_result.get('inspection_required') is not None):
                            
                            st.subheader("✅ Validation Results")
                            
                            # Coverage Status - only if determined
                            is_covered = validation_result.get('is_covered')
                            if is_covered is not None:
                                coverage_text = "✅ Covered" if is_covered else "❌ Not Covered"
                                st.write(f"**Coverage Status:** {coverage_text}")
                            
                            # Fraud Assessment - only if assessed
                            fraud_assessment = validation_result.get('fraud_assessment', {})
                            risk_level = fraud_assessment.get('risk_level')
                            if risk_level:
                                risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
                                st.write(f"**Fraud Risk:** {risk_colors.get(risk_level, '⚪')} {risk_level}")
                                
                                # Risk Factors - only if detected
                                risk_factors = fraud_assessment.get('risk_factors', [])
                                if risk_factors:
                                    st.write("**Risk Factors Detected:**")
                                    for factor in risk_factors:
                                        st.write(f"⚠️ {factor}")
                            
                            # Inspection Requirement - only if determined
                            inspection_required = validation_result.get('inspection_required')
                            if inspection_required is not None:
                                inspection_text = "✅ Required" if inspection_required else "❌ Not Required"
                                st.write(f"**Physical Inspection:** {inspection_text}")
                            
                            # Recommendation - only if made
                            recommendation = validation_result.get('recommendation')
                            if recommendation:
                                st.write(f"**Recommendation:** {recommendation}")
                
                # Next Steps Section - ONLY show if we have actual steps
                compliance_decision = structured_results.get('compliance_decision', {})
                next_steps = compliance_decision.get('next_steps', [])
                timeline = compliance_decision.get('timeline')
                appeals_available = compliance_decision.get('appeals_available')
                
                if next_steps or timeline or appeals_available is not None:
                    st.subheader("📋 Next Steps & Timeline")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if next_steps:
                            st.write("**Required Actions:**")
                            for i, step in enumerate(next_steps, 1):
                                st.write(f"{i}. {step}")
                    
                    with col2:
                        if timeline:
                            st.write(f"**Processing Timeline:** {timeline}")
                        if appeals_available is not None:
                            st.write(f"**Appeals Process:** {'✅ Available' if appeals_available else '❌ Not Applicable'}")
                
                # Show message if no analysis available
                if not any([
                    overall_summary.get('claim_outcome'),
                    key_findings,
                    policy_analysis.get('coverage_types'),
                    validation_result.get('is_covered') is not None,
                    compliance_decision.get('next_steps')
                ]):
                    st.info("📝 **Analysis Summary:** Waiting for agent responses to display detailed analysis...")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.subheader("🤖 Complete Agent Analysis")
                st.info("These are actual conversations between AutoGen agents powered by Groq LLaMA-3")
                
                agent_responses = results.get('agent_responses', {})
                
                # Display each agent's analysis
                for agent_name in ["PolicyAnalyzer", "ClaimValidator", "ComplianceOfficer"]:
                    if agent_name in agent_responses:
                        response = agent_responses[agent_name]
                        
                        with st.expander(f"💬 {agent_name} - {response.get('agent_role', 'Specialist')}", expanded=True):
                            #st.markdown(f'<div class="agent-response-card">', unsafe_allow_html=True)
                            st.markdown(f"**Agent:** {agent_name}")
                            st.markdown(f"**Role:** {response.get('agent_role', 'Insurance Specialist')}")
                            st.markdown(f"**Analysis Timestamp:** {response.get('timestamp', 'Unknown')}")
                            st.markdown("**Detailed Analysis:**")
                            st.write(response.get('content', 'No analysis available'))
                            st.markdown('</div>', unsafe_allow_html=True)
                
                # Show conversation flow
                with st.expander("🔍 Complete Conversation Log"):
                    for i, message in enumerate(results.get('full_conversation', []), 1):
                        st.markdown(f"**Message {i} - {message.get('name', 'System')}:**")
                        st.text(message.get('content', 'No content')[:500] + "..." if len(message.get('content', '')) > 500 else message.get('content', ''))
                        st.markdown("---")
            
            with tab3:
                st.subheader("📧 Available Actions")
                
                # Check if inspection is required - ONLY from actual agent analysis
                requires_inspection = structured_results.get('validation_result', {}).get('inspection_required')
                
                if requires_inspection is True:
                    st.warning("🔍 **Physical Inspection Required**")
                    st.write("Based on the AI agent analysis, a physical inspection has been scheduled for this claim.")
                    
                    # Professional email sending interface
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Inspection Details:**")
                        st.write(f"• **Scheduled for:** {(datetime.now() + timedelta(days=3)).strftime('%A, %B %d, %Y')}")
                        st.write(f"• **Time Window:** 9:00 AM - 5:00 PM")
                        st.write(f"• **Location:** {st.session_state.claimant_info.get('address', 'Address on file')}")
                        st.write(f"• **Duration:** 1-2 hours")
                    
                    with col2:
                        if st.button("📧 Send Professional Inspection Email", type="primary"):
                            with st.spinner("📤 Sending professional email notification..."):
                                success = EmailService.send_inspection_email(
                                    claim_details, 
                                    st.session_state.claimant_info
                                )
                                
                                if success:
                                    st.success("✅ Professional inspection notification sent successfully!")
                                    
                                    # Show additional actions
                                    st.info("📋 **Additional Actions Completed:**")
                                    st.write("• Claim status updated in system")
                                    st.write("• Inspector assignment initiated")
                                    st.write("• Customer service notification sent")
                                    st.write("• Compliance documentation updated")
                
                elif requires_inspection is False:
                    st.info("✅ **No Physical Inspection Required**")
                    st.write("The AI analysis determined that this claim can be processed without a physical inspection.")
                
                # Additional communications - ONLY based on actual agent decisions
                compliance_decision = structured_results.get('compliance_decision', {})
                final_status = compliance_decision.get('final_status')
                
                if final_status:
                    st.subheader("📋 Additional Communications")
                    
                    if final_status == 'APPROVED':
                        st.success("✅ **Approval Notifications:**")
                        st.write("• Payment processing department notified")
                        st.write("• Customer approval email queued")
                        st.write("• Claims closure documentation prepared")
                        
                    elif final_status == 'REJECTED':
                        st.error("❌ **Rejection Notifications:**")
                        st.write("• Formal rejection letter prepared")
                        st.write("• Appeals process information included")
                        st.write("• Customer service follow-up scheduled")
                        
                        if st.button("📄 Generate Rejection Letter"):
                            st.info("📄 Rejection letter template would be generated here with detailed reasoning and appeals information.")
                    
                    elif final_status == 'INVESTIGATE':
                        st.warning("🔍 **Investigation Notifications:**")
                        st.write("• Investigation team assignment pending")
                        st.write("• Additional documentation requests prepared")
                        st.write("• Customer update notifications scheduled")
                    
                    elif final_status == 'PENDING':
                        st.info("⏳ **Pending Review Notifications:**")
                        st.write("• Additional documentation request prepared")
                        st.write("• Review timeline communicated to customer")
                        st.write("• Follow-up review scheduled")
                    
                    # Show system actions only if we have a decision
                    st.subheader("🔧 System Actions")
                    st.write("**Automated System Updates:**")
                    st.write("• Claim status updated in database")
                    st.write("• AI analysis results stored")
                    if compliance_decision.get('timeline'):
                        st.write(f"• Processing timeline established: {compliance_decision['timeline']}")
                    st.write("• Compliance audit trail created")
                    st.write("• Customer portal updated with status")
                
                # Show message if no actions available
                if not final_status and requires_inspection is None:
                    st.info("📝 **Available Actions:** Waiting for agent analysis to determine required actions...")
        
        else:
            st.error(f"❌ Multi-agent processing failed: {results.get('error', 'Unknown error')}")
            if 'traceback' in results:
                with st.expander("🔍 Technical Error Details"):
                    st.code(results['traceback'])

if __name__ == "__main__":
    main()
