from langchain.prompts import ChatPromptTemplate


system_prompt = '''
You are a GDPR-compliant data privacy assistant. Your role is to detect Personally Identifiable Information (PII) in provided text based on the EU’s General Data Protection Regulation (GDPR).

PII includes any information relating to an identified or identifiable natural person, either directly (e.g., name, email address, national ID number) or indirectly (e.g., IP address, location data, unique device identifiers, or any data that can identify a person when combined with other information).

When analyzing documents, you must:
1. Identify and extract all PII instances.
2. Categorize each instance.
3. Determine if it is a direct or indirect identifier.
4. Justify the classification based on GDPR definitions.

Maintain strict compliance with GDPR’s definition of personal data as described in Article 4(1).

Output results in structured JSON array format, suitable for downstream processing.
If no PII is identified, return an empty JSON array.
'''

user_prompt_template = '''
### Instruction:
Analyze the following document and identify all instances of Personally Identifiable Information (PII) according to the EU's GDPR.

### Input:
{text}

### Response:
For each identified PII instance, return the:
- text: The extracted text, exactly as it appears in the document
- category: The PII category (e.g., name, email, phone number, IP address, health data)
- type: The PII identifier type (direct or indirect)
- justification: The justification for PII classification

Output results in structured JSON array format, where each object in the array represents a PII instance.
If no PII is identified, return an empty JSON array.

Example:
[
    {{"text": "John Doe", "category": "name", "type": "direct", "justification": "Identifies an individual directly."}},
    {{"text": "d.joe@brand.co", "category": "email", "type": "direct", "justification": "Identifies an individual directly through their email address."}}
]
'''

combination_prompt_template = '''
### Instruction:
You are given a list of structured JSON arrays of JSON objects, where each object in the array represents a Personally Identifiable Information (PII) instance.

Each PII JSON object contains the following attributes:
- text
- category
- type
- justification

Merge all arrays into a single final JSON array which contains all the JSON objects.

### Input:
{pii_lists}

### Response:
Output result in structured JSON array format, where each object in the array represents a PII instance.
'''


map_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", user_prompt_template),
    ]
)

reduce_prompt  = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", combination_prompt_template),
    ]
)