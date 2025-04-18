// static/js/assistant-services.js
const ASSISTANT_SERVICES = {
    ESSAY_ADVISOR: 'essay_advisor',
    K9_HELPER: 'k9_helper'
};

async function callAssistant(serviceName, prompt) {
    const response = await fetch(`/assistant/${serviceName}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt })
    });
    return await response.json();
}
