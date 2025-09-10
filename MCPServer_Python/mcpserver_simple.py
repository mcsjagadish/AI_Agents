from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP(name="Deloitte MCP Server", description="A single repository for all Deloitte MCP tools.")

# Constants
NWS_API_BASE = "https://deloitte.com"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    headers = {
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


@mcp.tool()
async def get_audit_guidelines(state: str) -> str:
    """Get audit guidelines for a US state.
    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/audit/guidelines/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch guidelines or no guidelines found."

    if not data["features"]:
        return "No audit guidelines available for this state."

    guidelines = []
    for feature in data["features"]:
        props = feature["properties"]
        guideline = f"""
        Title: {props.get('title', 'Unknown')}
        Description: {props.get('description', 'No description available')}
        URL: {props.get('url', 'No URL available')}
        """
        guidelines.append(guideline)

    return "\n---\n".join(guidelines)

@mcp.tool()
async def get_tax_alerts(state: str) -> str:
    """Get tax alerts for a US state.
    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = []
    for feature in data["features"]:
        props = feature["properties"]
        alert = f"""
        Event: {props.get('event', 'Unknown')}
        Area: {props.get('areaDesc', 'Unknown')}
        Severity: {props.get('severity', 'Unknown')}
        Description: {props.get('description', 'No description available')}
        Instructions: {props.get('instruction', 'No specific instructions provided')}
        """
        alerts.append(alert)

    return "\n---\n".join(alerts)

@mcp.tool()
async def get_tax_forms(state: str) -> str:
    """Get tax forms for a US state.
    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/forms/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch forms or no forms found."

    if not data["features"]:
        return "No tax forms available for this state."

    forms = []
    for feature in data["features"]:
        props = feature["properties"]
        form = f"""
        Form Name: {props.get('formName', 'Unknown')}
        Form Number: {props.get('formNumber', 'Unknown')}
        Description: {props.get('description', 'No description available')}
        URL: {props.get('url', 'No URL available')}
        """
        forms.append(form)

    return "\n---\n".join(forms)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='streaming_http', host='127.0.0.1', port=9000)
    #mcp.run(transport='stdio')
    