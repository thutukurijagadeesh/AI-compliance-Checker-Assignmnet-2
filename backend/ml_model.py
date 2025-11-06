import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None
from datasets import load_dataset
import pandas as pd

# Initialize OpenAI client
if OpenAI is not None:
    client = OpenAI(api_key="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")
else:
    client = None

# Initialize global instances early
ml_extractor = None
ml_risk_identifier = None

class MLClauseExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {
            'svm': SVC(kernel='linear', probability=True),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'nn': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        self.trained_models = {}
        self.bert_classifier = None

    def prepare_dataset(self):
        """Prepare dataset for training using CUAD or synthetic data with GDPR/HIPAA clauses."""
        try:
            # Try loading CUAD dataset
            dataset = load_dataset("cuad", split="train")
            texts = []
            labels = []

            for item in dataset:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict) and 'context' in item:
                    text = item['context']
                else:
                    text = str(item)
                # For simplicity, label based on presence of key terms
                label = 1 if any(term in text.lower() for term in [
                    'data processing', 'personal data', 'privacy', 'consent', 'breach',
                    'security', 'retention', 'transfer', 'rights', 'liability'
                ]) else 0
                texts.append(text)
                labels.append(label)

            return texts, labels
        except:
            # Fallback to synthetic data with GDPR/HIPAA clauses and non-contract examples
            print("CUAD dataset not available, using enhanced synthetic data with GDPR/HIPAA clauses")

            # GDPR Key Clauses (150)
            gdpr_clauses = [
                "Lawfulness of Processing – Processing must have a lawful basis under Article 6.",
                "Consent Requirement – Explicit consent required for certain processing activities.",
                "Data Minimization – Collect only necessary data for specific purposes.",
                "Purpose Limitation – Use personal data solely for declared purposes.",
                "Accuracy of Data – Ensure data accuracy and allow correction.",
                "Storage Limitation – Retain data only as long as necessary.",
                "Integrity and Confidentiality – Maintain data security and confidentiality.",
                "Accountability Principle – Controller responsible for demonstrating compliance.",
                "Data Protection by Design – Embed privacy principles into system design.",
                "Data Protection by Default – Default settings must favor privacy.",
                "Records of Processing Activities – Maintain documentation of processing operations.",
                "Data Subject Rights – Guarantee access, rectification, erasure, restriction, portability, and objection rights.",
                "Right to Access – Individuals can access their data.",
                "Right to Rectification – Individuals can correct inaccurate data.",
                "Right to Erasure – Individuals can request deletion of personal data.",
                "Right to Restrict Processing – Limit processing in certain cases.",
                "Right to Data Portability – Individuals can transfer their data to another controller.",
                "Right to Object – Individuals can object to processing.",
                "Automated Decision-Making – Restrictions on profiling and automation.",
                "Transparency Obligations – Provide clear information about processing.",
                "Privacy Notice Requirements – Inform users about purposes, rights, and controllers.",
                "Legal Basis – Contract – Processing necessary for performance of a contract.",
                "Legal Basis – Legal Obligation – Processing required by law.",
                "Legal Basis – Legitimate Interest – Processing justified by legitimate interests.",
                "Data Controller Obligations – Define controller responsibilities.",
                "Data Processor Obligations – Processor acts under documented instructions.",
                "Subprocessor Approval – Require controller approval for subcontractors.",
                "Cross-Border Data Transfers – Restrictions for data outside EEA.",
                "Standard Contractual Clauses – EU-approved data transfer terms.",
                "Binding Corporate Rules – Multinational group data transfer framework.",
                "EU–US Data Privacy Framework – Transfers to certified U.S. entities.",
                "Security of Processing – Implement technical and organizational measures.",
                "Data Breach Notification – Notify authorities within 72 hours.",
                "Data Breach Communication – Inform affected individuals if risk is high.",
                "Data Protection Impact Assessment – Assess risks for high-risk processing.",
                "Prior Consultation – Consult authority if DPIA shows residual risks.",
                "Appointment of DPO – Designate Data Protection Officer where required.",
                "DPO Independence – Ensure DPO operates independently.",
                "Cooperation with Supervisory Authority – Must cooperate with regulators.",
                "Data Processor Agreement – Contract between controller and processor.",
                "Confidentiality Obligations – Maintain data secrecy.",
                "Data Access Control – Restrict access to authorized users only.",
                "Data Encryption – Encrypt data at rest and in transit.",
                "Pseudonymization – Replace identifiers with pseudonyms.",
                "Anonymization – Irreversibly remove personal identifiers.",
                "Data Transfer Mechanisms – Ensure lawful transfer basis.",
                "Territorial Scope – Applies to entities inside and outside EU.",
                "Supervisory Authority Jurisdiction – One-stop-shop mechanism.",
                "Remedies and Liability – Right to lodge complaints and compensation.",
                "Fines and Penalties – Administrative fines up to €20 million.",
                "Joint Controller Arrangement – Transparency in shared responsibilities.",
                "Third-Party Disclosure – Govern data sharing with external entities.",
                "Children's Data Protection – Parental consent required under age 16.",
                "Consent Withdrawal – Mechanism for easy withdrawal.",
                "Profiling Limitations – Restrict automated profiling outcomes.",
                "Legitimate Interest Assessment – Document balancing test.",
                "Incident Response Plan – Define breach response protocol.",
                "Data Retention Schedule – Specify data retention periods.",
                "Vendor Due Diligence – Evaluate vendor compliance.",
                "Data Mapping – Maintain detailed record of data flows.",
                "Privacy Impact Assessment – Identify privacy risks early.",
                "Privacy Training – Conduct GDPR training for staff.",
                "Policy Review Clause – Review data policies periodically.",
                "Audit Rights – Allow controller to audit processors.",
                "Deletion on Termination – Processor must delete/return data.",
                "Subprocessor Liability – Primary processor responsible for subprocessors.",
                "Documentation Retention – Retain compliance evidence.",
                "Cooperation During Audit – Assist in regulatory inspections.",
                "Law Enforcement Disclosure – Handle authority requests properly.",
                "International Transfer Logs – Maintain cross-border transfer records.",
                "Data Subject Complaint Handling – Manage privacy complaints effectively.",
                "Privacy Governance Framework – Establish internal compliance structure.",
                "Cross-Border Group Policy – Common policy for group entities.",
                "Policy Breach Sanctions – Define disciplinary actions.",
                "Encryption Key Management – Manage cryptographic keys securely.",
                "Processor Subcontract Approval – Require written consent.",
                "Confidential Information Clause – Protect confidential data.",
                "Jurisdiction Clause – Define governing law and forum.",
                "Force Majeure – Handle unforeseen compliance disruptions.",
                "Termination Clause – Define termination rights and obligations.",
                "Data Restoration – Specify backup and recovery processes.",
                "Access Request Timeline – Respond to data subject requests within one month.",
                "Right to Lodge Complaint – Data subject's right to contact authority.",
                "Appoint EU Representative – For non-EU entities processing EU data.",
                "Non-Disclosure Agreement Link – Align NDAs with privacy rules.",
                "Technical and Organizational Measures – Describe implemented safeguards.",
                "Data Classification Policy – Define data sensitivity levels.",
                "Employee Confidentiality – Staff must sign confidentiality commitments.",
                "System Logging – Log system activity for accountability.",
                "Transfer Impact Assessment – Assess risk before data transfers.",
                "Privacy Notice Update – Keep notices current and clear.",
                "Customer Notification – Inform customers of policy changes.",
                "Data Sharing Agreement – Define obligations for shared data.",
                "Processor Indemnity – Processor liable for GDPR breaches.",
                "Controller Indemnity – Controller protected from processor's noncompliance.",
                "Record-Keeping Obligations – Maintain compliance documentation.",
                "Limitation of Liability – Define liability limits.",
                "Cross-Border Data Transfer Exceptions – Apply Article 49 derogations.",
                "Legitimate Interest Documentation – Evidence for justification.",
                "Security Testing – Conduct regular penetration tests.",
                "Privacy by Default Implementation – Apply minimal data collection by design.",
                "Data Subject Verification – Verify identity for requests.",
                "User Consent Tracking – Log consent given or withdrawn.",
                "Data Processing Location – Identify data hosting location.",
                "Vendor Contract Review – Ensure vendor contracts meet GDPR.",
                "Cross-Border Risk Mitigation – Manage transfer risk.",
                "Staff Access Review – Periodically review data access levels.",
                "Third-Party Risk Management – Evaluate third-party compliance.",
                "Policy Communication – Publish privacy policy publicly.",
                "Complaint Resolution Process – Internal procedure for GDPR issues.",
                "Secure Disposal Policy – Proper data deletion procedures.",
                "Encryption at Rest – Encrypt stored personal data.",
                "Encryption in Transit – Secure data transmission.",
                "Authentication Controls – Require strong authentication.",
                "Role-Based Access – Access based on job roles.",
                "Multi-Factor Authentication – Require MFA for sensitive systems.",
                "Data Transfer Documentation – Maintain legal basis record.",
                "Policy Awareness Campaign – Promote privacy awareness.",
                "Vendor Security Clause – Include mandatory security controls.",
                "Processor Auditing – Periodic review of processors.",
                "Incident Escalation Process – Define reporting steps.",
                "Breach Evidence Retention – Retain breach documentation.",
                "Forensic Investigation Clause – Allow forensic review post-breach.",
                "Supervisory Authority Liaison – Assign contact for DPA.",
                "Record Retention Period – Define retention per data category.",
                "Automated Processing Disclosure – Inform users of automated decisions.",
                "Consent Record Retention – Keep consent logs.",
                "Sensitive Data Protection – Enhanced protection for special categories.",
                "Employee Monitoring Disclosure – Transparency about monitoring.",
                "Transparency Register Entry – Record in national transparency registers.",
                "Policy Version Control – Version and timestamp data policies.",
                "Risk Assessment Review – Conduct periodic reassessment.",
                "Vendor Breach Notification – Vendors must notify of breaches.",
                "Privacy Certification Clause – Obtain privacy certifications.",
                "Data Lifecycle Management – Manage data from creation to deletion.",
                "Cloud Service Compliance – Cloud providers must comply with GDPR.",
                "Onboarding Privacy Review – Check privacy before new projects.",
                "Privacy Steering Committee – Internal oversight body.",
                "Data Flow Diagram Clause – Maintain updated data flow maps.",
                "Secure Coding Policy – Include privacy in code practices.",
                "Endpoint Protection Clause – Secure devices accessing data.",
                "Encryption Algorithm Standards – Define approved encryption types.",
                "Key Rotation Policy – Regularly rotate encryption keys.",
                "Password Policy Clause – Set minimum password standards.",
                "System Patch Management – Keep systems updated.",
                "Internal Audit Schedule – Conduct regular privacy audits.",
                "Privacy Awareness Clause – Require employee compliance awareness.",
                "Legal Hold Clause – Retain data for litigation if required.",
                "Third-Country Assessment Clause – Assess adequacy for transfers.",
                "Cross-Border Compliance Clause – Continuous compliance with transfer rules."
            ]

            # HIPAA Key Clauses (150)
            hipaa_clauses = [
                "Privacy Rule Compliance – Follow all 45 CFR Part 160 and 164 requirements.",
                "Security Rule Compliance – Protect ePHI confidentiality, integrity, and availability.",
                "Breach Notification Rule – Notify affected individuals within 60 days.",
                "Minimum Necessary Standard – Use the least PHI necessary.",
                "Business Associate Agreement – Define BA responsibilities.",
                "Permitted Uses and Disclosures – Specify allowed PHI uses.",
                "Authorization Requirement – Obtain authorization for nonstandard use.",
                "Administrative Safeguards – Define administrative security measures.",
                "Technical Safeguards – Define IT-related protections.",
                "Physical Safeguards – Protect facilities and devices.",
                "Risk Analysis – Identify threats to PHI.",
                "Risk Management – Mitigate identified security risks.",
                "Workforce Security – Authorize and supervise workforce access.",
                "Information Access Management – Restrict ePHI access appropriately.",
                "Security Awareness and Training – Educate workforce on HIPAA.",
                "Security Incident Procedures – Define reporting and response plan.",
                "Contingency Plan – Prepare for emergencies and system failures.",
                "Data Backup Plan – Maintain retrievable backup copies of ePHI.",
                "Disaster Recovery Plan – Procedures for restoring lost data.",
                "Emergency Mode Operation – Maintain operations during crisis.",
                "Evaluation Procedures – Periodically review compliance measures.",
                "Facility Access Controls – Restrict physical access.",
                "Workstation Security – Prevent unauthorized workstation access.",
                "Device and Media Controls – Manage media movement and disposal.",
                "Access Controls – Implement user authentication.",
                "Audit Controls – Log and review system activity.",
                "Integrity Controls – Prevent improper ePHI alteration.",
                "Person or Entity Authentication – Verify identity before access.",
                "Transmission Security – Protect PHI during electronic transmission.",
                "Encryption and Decryption – Secure ePHI using encryption.",
                "De-identification of PHI – Remove identifiers to exempt from HIPAA.",
                "Re-identification – Permit controlled re-linking.",
                "Breach Risk Assessment – Determine notification necessity.",
                "Breach Notification to HHS – Report breaches to regulators.",
                "Subcontractor Compliance – Ensure all subcontractors meet HIPAA.",
                "Sanction Policy – Disciplinary measures for violations.",
                "Policy Documentation – Maintain HIPAA documentation for six years.",
                "Policy Review – Review policies periodically.",
                "Notice of Privacy Practices – Provide clear NPP to patients.",
                "Patient Access Rights – Allow access to medical records.",
                "Amendment of PHI – Permit patient correction requests.",
                "Disclosure Accounting – Record disclosures.",
                "Confidential Communications – Provide alternate contact means.",
                "Restriction Requests – Allow limits on PHI sharing.",
                "Administrative Responsibility – Assign a compliance officer.",
                "Security Officer Designation – Assign security officer for ePHI.",
                "Training Documentation – Keep training records.",
                "Complaint Procedure – Provide mechanism for HIPAA complaints.",
                "Whistleblower Protection – Protect employees who report violations.",
                "Retaliation Prohibition – No retaliation against complainants.",
                "Business Associate Responsibilities – Define BA PHI handling.",
                "Indemnification Clause – Assign responsibility for violations.",
                "Termination of Agreement – Handle BA contract termination.",
                "Data Return or Destruction – Destroy or return PHI after contract ends.",
                "Use of Subcontractors – Require same safeguards downstream.",
                "Security Breach Reporting – Immediate notification required.",
                "Incident Logging – Maintain record of incidents.",
                "Device Encryption – Mandate encryption of portable devices.",
                "Email Encryption – Secure email containing PHI.",
                "Remote Access Security – Restrict remote connections.",
                "VPN Requirement – Use VPN for remote access.",
                "Password Policy – Define strong password standards.",
                "Automatic Logoff – Terminate inactive sessions.",
                "Access Termination – Revoke access upon role change or termination.",
                "Security Configuration Management – Apply secure configurations.",
                "Firewall Requirement – Implement network firewall.",
                "Antivirus Software – Require antivirus protection.",
                "Patch Management – Regularly update systems.",
                "Intrusion Detection – Implement intrusion monitoring.",
                "Physical Security Logs – Track facility entry.",
                "Visitor Authorization – Manage visitor access.",
                "Security Badge Requirement – Require ID badges.",
                "Screen Protection – Prevent screen visibility exposure.",
                "Paper Record Disposal – Shred or destroy PHI paper records.",
                "Data Retention Period – Define retention timeline.",
                "Backup Storage Security – Secure backup media.",
                "Offsite Storage Control – Secure offsite PHI storage.",
                "System Access Review – Regularly audit user access.",
                "Vendor Due Diligence – Vet third-party HIPAA compliance.",
                "Security Policy Awareness – Regular employee reminders.",
                "Breach Containment – Steps to contain data incidents.",
                "Encryption Key Management – Secure key lifecycle.",
                "Multi-Factor Authentication – Implement MFA.",
                "Role-Based Access Control – Assign access by job function.",
                "PHI Disclosure Log – Maintain ongoing disclosure record.",
                "Policy Distribution – Communicate updated policies to staff.",
                "Portable Device Policy – Restrict PHI on personal devices.",
                "Fax Transmission Security – Secure fax use.",
                "Medical Equipment Security – Protect connected devices.",
                "Vendor Breach Liability – Vendor responsible for breach costs.",
                "HHS Audit Cooperation – Cooperate with federal audits.",
                "State Law Preemption Clause – HIPAA overrides conflicting state laws.",
                "Training Frequency – Conduct annual HIPAA training.",
                "Access Request Timeline – Respond to patient requests within 30 days.",
                "Emergency Disclosure Procedure – Allow necessary disclosures.",
                "Business Continuity Plan – Ensure continuous operations.",
                "Annual Risk Assessment – Conduct yearly analysis.",
                "Security Awareness Posters – Visible security reminders.",
                "PHI Use for Research – Governed by HIPAA research rules.",
                "Marketing and Fundraising Restrictions – Limit PHI use.",
                "Employee Confidentiality Agreement – Require signed acknowledgment.",
                "Hybrid Entity Designation – Separate covered components.",
                "Organized Health Care Arrangement – Define shared arrangement.",
                "Disclosure for Legal Purposes – Governed by specific conditions.",
                "Law Enforcement Requests – Disclosure only per rules.",
                "Subpoena Response Policy – Verify legality before disclosure.",
                "Deceased Individuals Clause – Protection extends 50 years post-death.",
                "Data Integrity Monitoring – Detect unauthorized alteration.",
                "Log Retention – Maintain audit logs.",
                "Audit Log Review – Periodic review of access logs.",
                "Security Violation Escalation – Define escalation levels.",
                "Encryption Algorithm Standard – Define approved methods.",
                "Workstation Use Policy – Define acceptable use.",
                "Mobile Device Security – Require secure mobile management.",
                "Cloud Service Compliance – Require HIPAA-compliant hosting.",
                "Subcontractor BA Agreement – Required for each subcontractor.",
                "Security Policy Exception Handling – Formal process for exceptions.",
                "Threat Intelligence Monitoring – Watch for new vulnerabilities.",
                "Patch Validation Process – Verify patches before deployment.",
                "Annual Policy Acknowledgment – Staff re-sign policies yearly.",
                "Sanctions Enforcement – Apply discipline for noncompliance.",
                "Data Integrity Verification – Validate stored data integrity.",
                "Authentication Mechanism – Use secure authentication.",
                "Termination Certification – Certify destruction of PHI.",
                "Internal Audit Schedule – Conduct regular HIPAA audits.",
                "Policy Approval Authority – Assign approvers for policies.",
                "Workforce Accountability – Document workforce compliance.",
                "Vendor Access Limitation – Restrict vendor system access.",
                "Network Segmentation – Isolate sensitive systems.",
                "Security Patch Schedule – Define patch timelines.",
                "Annual Policy Review – Conduct yearly review.",
                "Password Expiration – Require periodic password changes.",
                "Data Loss Prevention – Implement DLP tools.",
                "Compliance Metrics – Track compliance KPIs.",
                "Mobile Encryption – Encrypt portable devices.",
                "Data Disposal Verification – Verify data destruction.",
                "Retention of Audit Records – Maintain audit logs for six years.",
                "Privacy Complaint Process – Handle privacy complaints promptly.",
                "HIPAA Certification Requirement – Encourage certification.",
                "BA Termination Clause – Termination for noncompliance.",
                "BA Notification Clause – Notify covered entity of incidents.",
                "Data Transmission Logging – Log all ePHI transmissions.",
                "Information System Activity Review – Regular review of audit logs.",
                "Downtime Procedures – Maintain PHI access during outages.",
                "Encryption Policy Documentation – Maintain encryption records.",
                "Password Complexity Rule – Enforce strong password policy.",
                "Remote Device Wipe – Allow remote wiping of lost devices.",
                "Internal Breach Investigation – Investigate incidents internally.",
                "Reporting Obligation – Promptly report security events.",
                "Compliance Attestation – Require formal compliance attestation annually."
            ]

            # Combine all contract clauses
            contract_texts = gdpr_clauses + hipaa_clauses

            # Non-contract examples
            non_contract_texts = [
                "This is a simple business letter about payment terms and invoice schedules.",
                "Dear Customer, thank you for your recent purchase. Your order has been processed.",
                "Please find attached the quarterly financial report for Q1 2023.",
                "Meeting agenda: Discuss project timeline and deliverables.",
                "Employee resume: John Doe, Software Engineer with 5 years experience.",
                "Product specifications: Model X features include wireless charging.",
                "Company newsletter: Updates on recent company events and achievements.",
                "Invoice #12345 for services rendered in the amount of $5000.",
                "Email correspondence regarding project status update.",
                "Technical documentation for API integration guidelines."
            ]

            # Create balanced dataset
            texts = contract_texts + non_contract_texts
            labels = [1] * len(contract_texts) + [0] * len(non_contract_texts)

            return texts, labels

    def train_models(self):
        """Train ML models on prepared dataset."""
        texts, labels = self.prepare_dataset()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"Training {name.upper()}...")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name.upper()} Accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred))

            self.trained_models[name] = model

        # Save models
        os.makedirs('models', exist_ok=True)
        for name, model in self.trained_models.items():
            with open(f'models/{name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)

        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_models(self):
        """Load trained models."""
        try:
            for name in self.models.keys():
                with open(f'models/{name}_model.pkl', 'rb') as f:
                    self.trained_models[name] = pickle.load(f)

            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)

            print("Models loaded successfully")
        except FileNotFoundError:
            print("Models not found, training new ones...")
            self.train_models()

    def predict_clauses_ml(self, text: str) -> dict:
        """Predict key clauses using ML models."""
        if not self.trained_models:
            self.load_models()

        text_vec = self.vectorizer.transform([text])

        results = {}
        for name, model in self.trained_models.items():
            prob = model.predict_proba(text_vec)[0][1]  # Probability of being key clause
            results[name] = prob

        # Use ensemble prediction (average probability)
        avg_prob = sum(results.values()) / len(results)
        is_key_clause = avg_prob > 0.5

        return {
            'is_key_clause': is_key_clause,
            'confidence': avg_prob,
            'model_predictions': results
        }

    def predict_clauses_bert(self, text: str) -> dict:
        """Predict key clauses using BERT."""
        if self.bert_classifier is None:
            self.bert_classifier = pipeline(
                "text-classification",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )

        # For simplicity, use sentiment as proxy for clause importance
        result = self.bert_classifier(text[:512])[0]  # Limit text length
        confidence = float(result['score'])

        return {
            'is_key_clause': confidence > 0.7,  # Threshold for positive sentiment
            'confidence': confidence,
            'label': result['label']
        }

    def predict_clauses_gpt(self, text: str) -> dict:
        """Predict key clauses using GPT."""
        if client is None:
            return {
                "is_key_clause": False,
                "confidence": 0.5,
                "reasoning": "OpenAI client not available"
            }

        prompt = f"""
        Analyze the following text and determine if it contains key compliance clauses related to data privacy, security, or legal obligations.
        Text: {text[:1000]}  # Limit text length

        Respond with JSON format:
        {{
            "is_key_clause": true/false,
            "confidence": 0-1,
            "reasoning": "brief explanation"
        }}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()
            # Parse JSON response
            import json
            parsed = json.loads(result)
            return parsed
        except Exception as e:
            print(f"GPT prediction error: {e}")
            return {
                "is_key_clause": False,
                "confidence": 0.5,
                "reasoning": "Error in GPT prediction"
            }

class MLRiskIdentifier:
    def __init__(self):
        self.ml_extractor = MLClauseExtractor()

    def identify_risks_ml(self, text: str) -> list:
        """Identify risks using ML predictions."""
        # Split text into sentences and predict for each
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        risks = []
        for sent in doc.sents:
            pred = self.ml_extractor.predict_clauses_ml(sent.text)
            if pred['is_key_clause']:
                # Map to risk categories with standardized Low/Medium/High levels
                risk_mapping = {
                    'data processing': {'description': 'GDPR violation - processing personal data without lawful basis', 'severity': 'High'},
                    'personal data': {'description': 'GDPR risk - personal data handling', 'severity': 'Medium'},
                    'privacy': {'description': 'Privacy risk - potential unauthorized access', 'severity': 'Medium'},
                    'consent': {'description': 'GDPR requirement - explicit consent needed', 'severity': 'High'},
                    'breach': {'description': 'GDPR/HIPAA requirement - breach notification', 'severity': 'High'},
                    'security': {'description': 'Security risk - inadequate safeguards', 'severity': 'High'},
                    'retention': {'description': 'GDPR risk - data retention policies', 'severity': 'Medium'},
                    'transfer': {'description': 'GDPR risk - international data transfers', 'severity': 'Medium'},
                    'rights': {'description': 'GDPR requirement - data subject rights', 'severity': 'Medium'},
                    'liability': {'description': 'Legal risk - liability clauses', 'severity': 'Low'}
                }

                sent_lower = sent.text.lower()
                for term, risk_info in risk_mapping.items():
                    if term in sent_lower:
                        risks.append({
                            'risk': term,
                            'description': risk_info['description'],
                            'severity': risk_info['severity'],
                            'confidence': pred['confidence']
                        })
                        break

        return risks[:10]

    def identify_risks_gpt(self, text: str) -> list:
        """Identify risks using GPT."""
        # Fallback to empty list since GPT API is not available
        print("GPT not available, skipping GPT risk identification")
        return []

    def classify_document_type(self, text: str) -> dict:
        """Classify document type using GPT for better accuracy."""
        if client is None:
            # Fallback to rule-based classification
            text_lower = text.lower()

            # Contract indicators (expanded list)
            contract_keywords = [
                'agreement', 'contract', 'parties', 'hereby', 'shall', 'obligations',
                'terms', 'conditions', 'governing law', 'jurisdiction', 'termination',
                'confidentiality', 'intellectual property', 'liability', 'indemnification',
                'breach', 'remedies', 'force majeure', 'severability', 'entire agreement',
                'amendment', 'waiver', 'notices', 'counterparts', 'execution',
                'data processing', 'personal data', 'privacy', 'consent', 'security',
                'retention', 'transfer', 'rights', 'breach notification', 'compliance',
                'gdpr', 'hipaa', 'data protection', 'privacy policy', 'data subject',
                'controller', 'processor', 'subprocessor', 'data transfer', 'encryption',
                'audit', 'certification', 'regulation', 'legal', 'binding', 'effective date',
                'signatory', 'witness', 'executed', 'in witness whereof', 'whereas'
            ]

            # Non-contract indicators
            non_contract_keywords = [
                'dear', 'thank you', 'please find attached', 'meeting agenda',
                'invoice', 'receipt', 'newsletter', 'memo', 'report', 'resume',
                'product specifications', 'technical documentation', 'email',
                'correspondence', 'update', 'announcement', 'bulletin'
            ]

            contract_score = sum(1 for keyword in contract_keywords if keyword in text_lower)
            non_contract_score = sum(1 for keyword in non_contract_keywords if keyword in text_lower)

            # More lenient logic: assume contract if contract_score >= 3 or significantly higher than non-contract
            is_contract = contract_score >= 3 or (contract_score > non_contract_score and contract_score >= 1)
            confidence = min(1.0, max(0.0, contract_score / 5.0))  # Scale confidence differently

            return {
                "is_contract": is_contract,
                "document_type": "contract" if is_contract else "non-contract",
                "confidence": confidence,
                "reasoning": f"Rule-based classification: contract_score={contract_score}, non_contract_score={non_contract_score}"
            }

        # Use GPT for classification
        prompt = f"""
        Analyze the following document text and classify its type. Focus on legal/compliance documents.

        Document Text (first 2000 characters):
        {text[:2000]}

        Classify the document into one of these categories:
        1. Data Processing Agreement (DPA)
        2. Privacy Policy
        3. Terms of Service
        4. Business Associate Agreement (BAA)
        5. Data Protection Agreement
        6. General Contract
        7. Non-contract document

        Respond with JSON format:
        {{
            "document_type": "category_name",
            "is_contract": true/false,
            "confidence": 0-1,
            "reasoning": "brief explanation of classification"
        }}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()
            import json
            parsed = json.loads(result)
            return parsed
        except Exception as e:
            print(f"GPT classification error: {e}")
            # Fallback to rule-based
            return self.classify_document_type(text)  # Recursive call to rule-based

# Initialize global instances
def initialize_ml_models():
    global ml_extractor, ml_risk_identifier
    if ml_extractor is None:
        ml_extractor = MLClauseExtractor()
    if ml_risk_identifier is None:
        ml_risk_identifier = MLRiskIdentifier()

# Initialize on import
initialize_ml_models()
