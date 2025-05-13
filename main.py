import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging
import time
import json
import sys
from typing import Any, Dict, Optional, List, Generator, Union
import httpx
from pydantic import BaseModel, Field, ValidationError
from collections import defaultdict
import base64
import mimetypes

st.cache_data.clear()


# --- Configuration & Setup ---
st.set_page_config(layout="wide", page_title="Halo Ticket Interface")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Keep httpx logs quieter

# --- Load Configuration & Password from st.secrets ---
try:
    APP_PASSWORD = st.secrets.get("app_config", {}).get("password")
    HALO_TENANT = st.secrets["halo_credentials"]["tenant"]
    HALO_CLIENT_ID = st.secrets["halo_credentials"]["client_id"]
    HALO_SECRET = st.secrets["halo_credentials"]["secret"]
    HALO_CUSTOMER_CLIENT_ID = st.secrets.get("halo_credentials", {}).get(
        "customer_client_id"
    )
    if HALO_CUSTOMER_CLIENT_ID is not None:  # Ensure it's an int if provided
        HALO_CUSTOMER_CLIENT_ID = int(HALO_CUSTOMER_CLIENT_ID)

    DEFAULT_AGENT_ID = int(st.secrets.get("app_config", {}).get("default_agent_id", 21))
    APP_TIMEZONE_STR = st.secrets.get("app_config", {}).get(
        "timezone", "Australia/Perth"
    )

    if not all([HALO_TENANT, HALO_CLIENT_ID, HALO_SECRET]):
        st.error(
            "Error: Missing required Halo credentials in Streamlit secrets.", icon="🚨"
        )
        log.error("Missing required Halo credentials in st.secrets")
        st.stop()
    if not APP_PASSWORD:
        log.warning("App password not set in secrets. Access will be open.")

except KeyError as e:
    st.error(
        f"Error: Missing secret key: {e}. Configure secrets in Streamlit Cloud.",
        icon="🔒",
    )
    log.error(f"Missing secret key in st.secrets: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading secrets: {e}", icon="❓")
    log.error(f"Error loading secrets: {e}", exc_info=True)
    st.stop()

# --- Timezone Setup ---
try:
    TIMEZONE = pytz.timezone(APP_TIMEZONE_STR)
except pytz.UnknownTimeZoneError:
    log.warning(f"Timezone '{APP_TIMEZONE_STR}' from secrets not found. Using UTC.")
    TIMEZONE = pytz.utc

# --- Constants ---
TICKET_STATUS_CLOSED = 9
DEFAULT_NEW_STATUS_ID = 1
DEFAULT_TEAM_NAME = ""
DEFAULT_IMPACT = 3
DEFAULT_URGENCY = 2
DEFAULT_TICKET_TYPE_ID = 1
MAX_SEARCH_RESULTS_TO_DISPLAY = 10  # Changed to 10
MAX_PAGES_FOR_SEARCH = 100  # Increased for broader search (e.g., 3000 tickets)
MAX_PAGES_FOR_ALL_OPEN = 500  # For the main "All" tab (e.g., 5000 tickets)
# MAX_PAGES_FOR_TEAM_TICKETS not used with simpler team tab logic


# --- Password Protection Function ---
def check_password():
    if not APP_PASSWORD:
        return True
    if st.session_state.get("password_correct", False):
        return True
    st.header("🔒 App Locked")
    st.write("Please enter the password to access the Halo Ticket Interface.")
    with st.form("password_form"):
        password_attempt = st.text_input(
            "Password", type="password", key="password_input_field"
        )
        submitted = st.form_submit_button("Login")
        if submitted:
            if password_attempt == APP_PASSWORD:
                st.session_state["password_correct"] = True
                log.info("Password correct.")
                st.rerun()
            else:
                st.error("Password incorrect", icon="🚨")
                st.session_state["password_correct"] = False
                st.stop()
        else:
            st.stop()
    if not st.session_state.get("password_correct", False):
        st.stop()
    return st.session_state.get("password_correct", False)


if not check_password():
    sys.exit()


# --- Inlined Models (Same as before) ---
class HaloBaseModel(BaseModel):
    class Config:
        extra = "ignore"

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        if hasattr(super(), "dict"):
            return super().dict(*args, **kwargs)  # type: ignore
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args, **kwargs) -> str:
        kwargs.setdefault("exclude_none", True)
        if hasattr(super(), "json"):
            return super().json(*args, **kwargs)  # type: ignore
        return super().model_dump_json(*args, **kwargs)


class HaloRef(HaloBaseModel):
    id: int | str | None = None
    name: Optional[str] = None


class User(HaloBaseModel):
    id: int
    name: str | None = None
    site_id: int | None = None
    site_name: str | None = None
    client_name: str | None = None
    emailaddress: str | None = None
    inactive: bool = False


class Site(HaloBaseModel):
    id: int
    name: str
    client_id: int | None = None
    client_name: str | None = None
    inactive: bool = False


class Category(HaloBaseModel):
    id: int
    category_name: str


class Ticket(HaloBaseModel):
    id: int | None = None
    summary: str | None = None
    status_id: HaloRef | None = Field(
        default_factory=lambda: HaloRef(id=DEFAULT_NEW_STATUS_ID)
    )
    site_id: HaloRef | None = None
    site_name: str | None = None
    user_id: int | None = None
    user_name: str | None = None
    team: str | None = None
    agent_id: int | None = None
    categoryid_1: HaloRef | None = None
    last_update: str | None = None
    user_email: str | None = None
    hasbeenclosed: bool = False
    dateclosed: str | None = None
    impact: int | None = None
    urgency: int | None = None
    use: str | None = "ticket"
    details: str | None = None
    details_html: str | None = None
    customfields: Optional[List[Dict[str, Any]]] = None


class TicketAction(HaloBaseModel):
    id: int = 0
    ticket_id: int | None = None
    datetime: str | None = None
    note: str | None = None
    note_html: str | None = None
    outcome: str | None = None
    who: str | None = None
    createdby: str | None = None
    emailto: str | None = None
    emailcc: str | None = None
    emailfrom: str | None = None
    subject: str | None = None


class Status(HaloBaseModel):
    id: int
    name: str


# --- Inlined HaloClient ---
class StandaloneHaloClient:  # ... (Most methods same as before, upload_attachment and team ticket fetching adjusted)
    def __init__(self) -> None:
        self.secret: str = HALO_SECRET
        self.client_id: str = HALO_CLIENT_ID
        self.customer_client_id: Optional[int] = HALO_CUSTOMER_CLIENT_ID
        self.tenant: str = HALO_TENANT
        self.api_host_url: str = f"https://{self.tenant}.haloitsm.com"
        self.api_auth_url: str = f"{self.api_host_url}/auth/token"
        self.api_url: str = f"{self.api_host_url}/api"
        self._client: httpx.Client = httpx.Client(
            headers={"Accept": "application/json"}, timeout=60.0
        )
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        log.info(f"Halo Client configured for tenant: {self.tenant} (using st.secrets)")

    def _auth(self, grant_type="client_credentials"):  # ... (Same as before)
        body = {
            "grant_type": grant_type,
            "client_id": self.client_id,
            "client_secret": self.secret,
            "scope": ["all"],
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        try:
            response = httpx.post(url=self.api_auth_url, data=body, headers=headers)
            response.raise_for_status()
            creds = response.json()
            self._access_token = creds.get("access_token")
            if not self._access_token:
                raise ValueError("No access token received.")
            expires_in = creds.get("expires_in", 3600)
            self._token_expiry = datetime.now(pytz.utc) + timedelta(
                seconds=expires_in - 60
            )
            self._client.headers["Authorization"] = f"Bearer {self._access_token}"
        except Exception as e:
            log.error(f"Halo auth error: {e}", exc_info=True)
            raise ConnectionError(f"Halo Auth Failed: {e}") from e

    def _check_token(self):  # ... (Same as before)
        now_utc = datetime.now(pytz.utc)
        if (
            not self._access_token
            or not self._token_expiry
            or now_utc >= self._token_expiry
        ):
            log.info("Halo token refresh needed.")
            self._auth()
        if not self._access_token:
            raise ConnectionError("Failed to obtain/refresh Halo access token.")

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Any] = None,
    ) -> httpx.Response:  # ... (Same as before)
        self._check_token()
        url = f"{self.api_url}{endpoint}"
        headers = self._client.headers.copy()
        try:
            response = self._client.request(
                method, url, params=params, json=json_data, headers=headers
            )
            if response.status_code == 401:
                log.warning("Retrying after 401 Unauthorized...")
                self._access_token = None
                self._token_expiry = None
                self._auth()
                headers["Authorization"] = f"Bearer {self._access_token}"
                response = self._client.request(
                    method, url, params=params, json=json_data, headers=headers
                )
            is_ticket_post_or_create = (
                method.upper() == "POST"
            ) and endpoint.startswith("/tickets")
            if not is_ticket_post_or_create or response.status_code != 400:
                response.raise_for_status()
            return response
        except httpx.RequestError as e:
            log.error(f"Network error: {url}: {e}", exc_info=True)
            raise ConnectionError(f"Network error: {e}") from e
        except httpx.HTTPStatusError as e:
            log.error(
                f"HTTP error ({e.response.status_code}) for {url}: {e.response.text}",
                exc_info=True,
            )
            if e.response.status_code == 404:
                raise FileNotFoundError(f"Resource not found: {url}") from e
            if e.response.status_code == 403:
                raise PermissionError(f"Permission denied for {url}.") from e
            raise ConnectionError(
                f"HTTP error {e.response.status_code}: {e.response.text}"
            ) from e
        except Exception as e:
            log.error(f"Unexpected error: {url}: {e}", exc_info=True)
            raise RuntimeError(f"Failed request: {e}") from e

    def _get(
        self, endpoint: str, params: Optional[Dict] = None
    ) -> Any:  # ... (Same as before)
        response = self._request("GET", endpoint, params=params)
        try:
            return response.json()
        except json.JSONDecodeError:
            log.error(f"Bad JSON GET {endpoint}: {response.text}")
            raise ValueError(f"Bad JSON GET {endpoint}")

    def _post(self, endpoint: str, data: Any) -> Any:  # ... (Same as before)
        response = self._request("POST", endpoint, json_data=data)
        is_ticket_create = endpoint.startswith("/tickets")
        if is_ticket_create and response.status_code == 400:
            log.warning(f"POST {endpoint} 400 Bad Request. Resp: {response.text}")
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text
        try:
            if response.status_code == 204:
                return None
            return response.json()
        except json.JSONDecodeError:
            log.error(
                f"Bad JSON POST {endpoint} (Status: {response.status_code}): {response.text}"
            )
            return {
                "error": f"Bad JSON (Status: {response.status_code})",
                "response_text": response.text,
            }

    def _get_paged(
        self, endpoint: str, params: Optional[Dict] = None, max_pages=100
    ) -> Generator[Dict, Any, None]:  # ... (Same as before)
        if params is None:
            params = {}
        page_size = params.get("page_size", 100)
        params["page_size"] = page_size
        params["page_no"] = 1
        pages_fetched = 0
        total_items_yielded = 0
        while pages_fetched < max_pages:
            pages_fetched += 1
            try:
                data = self._get(endpoint, params)
                items = []
                record_count = 0
                items_on_page = 0
                if isinstance(data, list):
                    items = data
                    record_count = len(items)
                    items_on_page = len(items)
                elif isinstance(data, dict):
                    record_count = data.get("record_count", 0)
                    list_key = endpoint.split("/")[-1].lower()
                    if list_key in data and isinstance(data[list_key], list):
                        items = data[list_key]
                    else:
                        key = next(
                            (k for k, v in data.items() if isinstance(v, list)), None
                        )
                        items = data.get(key, []) if key else []
                    items_on_page = len(items)
                else:
                    log.warning(
                        f"Unexpected data type {type(data)} for {endpoint} page {params['page_no']}."
                    )
                    break
                if not items:
                    break
                yield from items
                total_items_yielded += items_on_page
                if record_count > 0 and total_items_yielded >= record_count:
                    break
                elif items_on_page < page_size:
                    break
                params["page_no"] += 1
                time.sleep(0.05)
            except Exception as e:
                log.error(
                    f"Error fetching page {params['page_no']} for {endpoint}: {e}",
                    exc_info=True,
                )
                break
        if pages_fetched >= max_pages:
            log.warning(f"Max pages ({max_pages}) hit for {endpoint}.")

    def _get_all_paged(
        self, endpoint: str, params: Optional[Dict] = None, max_pages=100
    ) -> List[Dict]:  # ... (Same as before)
        return list(self._get_paged(endpoint, params, max_pages=max_pages))

    def get_statuses(self) -> list[Status]:  # ... (Same as before)
        data = self._get("/status")
        items = data.get("status", []) if isinstance(data, dict) else data
        return [Status(**s) for s in items]

    def get_categories(self) -> list[Category]:  # ... (Same as before)
        params = {}
        if self.customer_client_id:
            params["client_id"] = str(self.customer_client_id)
        items = self._get_all_paged("/category", params)
        return [Category(**c) for c in items if c]

    def get_users(self) -> list[User]:  # ... (Same as before)
        params = {"includeinactive": "false"}
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        items = self._get_all_paged("/users", params)
        return [User(**u) for u in items if u]

    def get_teams(self) -> list[dict]:  # ... (Same as before)
        params = {"includeinactive": "false"}
        return self._get_all_paged("/team", params)

    def get_sites(self) -> list[Site]:  # ... (Same as before)
        params = {}
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        items = self._get_all_paged("/site", params)
        return [Site(**s) for s in items if s]

    def get_ticket_types(self) -> list[dict]:  # ... (Same as before)
        return self._get_all_paged("/tickettype")

    def get_user_by_email(self, email: str) -> Optional[User]:  # ... (Same as before)
        if not email:
            return None
        params = {"emailaddress": email, "includeinactive": "true"}
        if self.customer_client_id:
            params["client_id"] = str(self.customer_client_id)
        users_data = list(self._get_paged("/users", params, max_pages=1))
        if users_data:
            try:
                for user_data in users_data:
                    if user_data.get("emailaddress", "").lower() == email.lower():
                        return User(**user_data)
                return None
            except Exception as e:
                log.error(f"Error parsing user data for email {email}: {e}")
                return None
        return None

    def get_ticket(self, ticket_id: int) -> Optional[Ticket]:  # ... (Same as before)
        if not ticket_id:
            return None
        params = {
            "includedetails": True,
            "includeCustomFields": True,
            "includeHtmlDetails": True,
        }
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        try:
            data = self._get(f"/tickets/{ticket_id}", params)
            if isinstance(data.get("status_id"), int):
                data["status_id"] = {"id": data["status_id"]}
            elif data.get("status_id") is None:
                data["status_id"] = HaloRef(id=DEFAULT_NEW_STATUS_ID)
            if isinstance(data.get("site_id"), int):
                data["site_id"] = {"id": data["site_id"]}
            else:
                data["site_id"] = data.get("site_id")
            if isinstance(data.get("categoryid_1"), int):
                data["categoryid_1"] = {"id": data["categoryid_1"]}
            else:
                data["categoryid_1"] = data.get("categoryid_1")
            cf_key = next(
                (k for k in ["customfields", "customFields"] if k in data), None
            )
            if cf_key:
                data["customfields"] = data.pop(cf_key)
            else:
                data["customfields"] = []
            return Ticket(**data)
        except FileNotFoundError:
            log.warning(f"Ticket {ticket_id} not found.")
            return None
        except Exception as e:
            log.error(f"Failed to get/parse ticket {ticket_id}: {e}", exc_info=True)
            return None

    def _parse_raw_ticket_data(
        self, data: dict
    ) -> Optional[Ticket]:  # ... (Same as before)
        ticket_id_for_log = data.get("id", "N/A")
        try:
            for field_name in ["status_id", "site_id", "categoryid_1"]:
                field_val = data.get(field_name)
                if isinstance(field_val, int):
                    data[field_name] = HaloRef(id=field_val)
                elif isinstance(field_val, dict) and "id" in field_val:
                    data[field_name] = HaloRef(**field_val)
                elif field_val is None and field_name == "status_id":
                    data[field_name] = HaloRef(id=DEFAULT_NEW_STATUS_ID)
                elif field_val is None:
                    data[field_name] = None
            if "summary" not in data or data["summary"] is None:
                data["summary"] = "No Summary Provided"
            if "use" not in data:
                data["use"] = "ticket"
            if "team" not in data:
                data["team"] = None
            ticket = Ticket(**data)
            if ticket.status_id and ticket.status_id.id == TICKET_STATUS_CLOSED:
                return None
            return ticket
        except ValidationError as val_err:
            log.warning(
                f"Validation Error parsing ticket {ticket_id_for_log}: {val_err}"
            )
            return None
        except Exception as parse_error:
            log.warning(
                f"General Error parsing ticket {ticket_id_for_log}: {parse_error}"
            )
            return None

    def get_all_open_tickets(
        self,
    ) -> list[Ticket]:  # Uses constant MAX_PAGES_FOR_ALL_OPEN
        params = {
            "excludeclosed": "true",
            "orderby": "id",
            "orderbydesc": "true",
            "includeCustomFields": "false",
            "includedetails": "false",
            "page_size": 100,
        }
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        parsed_tickets = []
        try:
            for data in self._get_paged(
                "/tickets", params, max_pages=MAX_PAGES_FOR_ALL_OPEN
            ):
                ticket = self._parse_raw_ticket_data(data)
                if ticket:
                    parsed_tickets.append(ticket)
        except PermissionError as pe:
            log.error(f"Permission denied fetching all tickets: {pe}")
            return []
        except Exception as e:
            log.error(f"Failed to fetch all open tickets: {e}", exc_info=True)
            return []
        return parsed_tickets

    def get_open_tickets_for_team(
        self, team_id: int
    ) -> list[Ticket]:  # Uses constant MAX_PAGES_FOR_TEAM_TICKETS
        params = {
            "team_id": team_id,
            "excludeclosed": "true",
            "orderby": "id",
            "orderbydesc": "true",
            "includeCustomFields": "false",
            "includedetails": "false",
            "page_size": 100,
        }
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        parsed_tickets = []
        try:
            for data in self._get_paged(
                "/tickets", params, max_pages=MAX_PAGES_FOR_TEAM_TICKETS
            ):
                ticket = self._parse_raw_ticket_data(data)
                if ticket:
                    parsed_tickets.append(ticket)
        except Exception as e:
            log.error(
                f"Failed to fetch tickets for team ID {team_id}: {e}", exc_info=True
            )
        return parsed_tickets

    def search_tickets(
        self, search_term: str
    ) -> list[Ticket]:  # Uses constant MAX_PAGES_FOR_SEARCH
        if not search_term or len(search_term.strip()) < 3:
            log.info("Search: Term too short. Returning empty list.")
            return []
        endpoint = "/tickets"
        params = {
            "search": search_term,
            "excludeclosed": "true",
            "orderby": "id",
            "orderbydesc": "true",
            "includeCustomFields": "false",
            "includedetails": "false",
            "page_size": 100,
        }
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        parsed_tickets = []
        try:
            for data in self._get_paged(
                endpoint, params, max_pages=MAX_PAGES_FOR_SEARCH
            ):
                ticket = self._parse_raw_ticket_data(data)
                if ticket:
                    parsed_tickets.append(ticket)
            return parsed_tickets
        except FileNotFoundError:
            log.error(f"SEARCH: Endpoint {endpoint} not found.")
            st.toast(f"Search Error: API endpoint not found.", icon="🚫")
            return []
        except Exception as e:
            log.error(f"SEARCH: Failed for term '{search_term}': {e}", exc_info=True)
            st.toast(f"Search Error: {str(e)[:100]}", icon="🔥")
            return []

    def get_ticket_actions(
        self, ticket_id: int
    ) -> list[TicketAction]:  # ... (Same as before)
        if not ticket_id:
            return []
        params = {
            "ticket_id": ticket_id,
            "includeattachments": True,
            "includeagentdetails": True,
            "includehtmlnote": True,
        }
        actions_data = list(self._get_paged("/actions", params, max_pages=100))
        parsed_actions = []
        for a in actions_data:
            try:
                parsed_actions.append(TicketAction(**a))
            except Exception as parse_err:
                log.error(
                    f"Error parsing action {a.get('id')} for ticket {ticket_id}: {parse_err}"
                )
        return parsed_actions

    def update_ticket_field(
        self, ticket_id: int, field_name: str, new_value: Any
    ) -> bool:  # ... (Same as before)
        if not self.customer_client_id:
            log.error("Cannot update: HALO_CUSTOMER_CLIENT_ID not set.")
            st.error("Update failed: App config missing.")
            return False
        if field_name in ["status_id", "site_id", "categoryid_1"] and isinstance(
            new_value, int
        ):
            processed_value = new_value
        elif field_name == "agent_id" and isinstance(new_value, int):
            processed_value = new_value if new_value != 0 else None
        elif field_name == "team":
            processed_value = new_value if new_value else None
        else:
            processed_value = new_value
        payload = {
            "id": ticket_id,
            field_name: processed_value,
            "use": "ticket",
            "client_id": self.customer_client_id,
        }
        try:
            endpoint = f"/tickets?client_id={self.customer_client_id}"
            response = self._post(endpoint, [payload])
            if (
                response is None
                or (
                    isinstance(response, list)
                    and response
                    and response[0].get("id") == ticket_id
                )
                or (isinstance(response, dict) and response.get("id") == ticket_id)
            ):
                return True
            else:
                log.error(
                    f"Update '{field_name}' failed for ticket {ticket_id}. Response: {response}"
                )
                return False
        except Exception as e:
            log.error(
                f"Error updating '{field_name}' for ticket {ticket_id}: {e}",
                exc_info=True,
            )
            return False

    def upload_attachment_and_get_token(
        self, ticket_id: int, file_name: str, file_content: bytes
    ) -> Optional[str]:
        if not self.customer_client_id:
            log.error("Cannot upload: HALO_CUSTOMER_CLIENT_ID not set.")
            return None
        b64_content = base64.b64encode(file_content).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(file_name)
        if not mime_type:
            mime_type = "application/octet-stream"

        # Standard payload for Halo attachment upload.
        # The key is `data_base64` which should just be the base64 string.
        # `isimage` helps Halo identify it for embedding.
        payload_item = {
            "ticket_id": ticket_id,
            "filename": file_name,
            "data_base64": b64_content,  # Just the base64 content, not a data URL
            "isimage": True,  # Crucial for Halo to treat it as embeddable
            "filesize": len(file_content),
        }
        log.info(
            f"Uploading attachment '{file_name}' to ticket {ticket_id}. Mime type: {mime_type}"
        )
        try:
            response = self._post("/attachment", [payload_item])
            log.info(
                f"Attachment upload response for '{file_name}': {json.dumps(response, indent=2)}"
            )
            attachment_data = None
            if response and isinstance(response, list) and response[0]:
                attachment_data = response[0]
            elif response and isinstance(response, dict):
                attachment_data = response
            else:
                log.error(
                    f"Upload for '{file_name}' failed. Unexpected response: {response}"
                )
                return None

            if (
                attachment_data.get("id")
                and attachment_data.get("unique_id")
                and attachment_data.get("unique_id") != 0
            ):
                unique_id = attachment_data["unique_id"]
                log.info(
                    f"Attachment '{file_name}' uploaded. ID: {attachment_data.get('id')}, Unique ID for embedding: {unique_id}"
                )
                return f"/api/Attachment/image?token={unique_id}"  # Relative URL for Halo notes
            else:
                log.error(
                    f"Upload for '{file_name}' (ID: {attachment_data.get('id')}) returned unique_id: {attachment_data.get('unique_id')}. Cannot reliably embed."
                )
                return None  # Indicate failure to get a usable token
        except Exception as e:
            log.error(f"Error uploading attachment '{file_name}': {e}", exc_info=True)
            return None

    def add_ticket_action(
        self, payload: Dict[str, Any]
    ) -> bool:  # ... (Same as before)
        try:
            response = self._post("/actions", [payload])
            if response and (
                (isinstance(response, list) and response[0].get("id") is not None)
                or (isinstance(response, dict) and response.get("id") is not None)
            ):
                return True
            else:
                log.error(
                    f"Add action failed for ticket {payload.get('ticket_id')}. Response: {response}"
                )
                return False
        except Exception as e:
            log.error(
                f"Error adding action to ticket {payload.get('ticket_id')}: {e}",
                exc_info=True,
            )
            return False

    def close_ticket_with_agent(
        self, ticket_id: int, agent_id: Optional[int] = None
    ) -> bool:  # ... (Same as before)
        agent_to_use = agent_id if agent_id is not None else DEFAULT_AGENT_ID
        if not self.customer_client_id:
            log.error("Cannot close: HALO_CUSTOMER_CLIENT_ID not set.")
            st.error("Close failed: App config missing.")
            return False
        endpoint = f"/tickets?client_id={self.customer_client_id}"
        try:
            current_ticket = self.get_ticket(ticket_id)
            if not current_ticket:
                log.error(f"Cannot close ticket {ticket_id}: Failed to fetch.")
                return False
            current_agent_id = current_ticket.agent_id
            needs_agent_assign = current_agent_id is None or current_agent_id <= 1
            final_agent_id_for_closure = (
                agent_to_use if needs_agent_assign else current_agent_id
            )
            if needs_agent_assign:
                assign_payload = [
                    {
                        "id": ticket_id,
                        "agent_id": agent_to_use,
                        "use": "ticket",
                        "client_id": self.customer_client_id,
                    }
                ]
                try:
                    self._post(endpoint, assign_payload)
                    time.sleep(0.5)
                except Exception as assign_e:
                    log.warning(f"Pre-closure agent assign failed: {assign_e}")
            close_payload = [
                {
                    "id": ticket_id,
                    "status_id": TICKET_STATUS_CLOSED,
                    "agent_id": final_agent_id_for_closure,
                    "use": "ticket",
                    "client_id": self.customer_client_id,
                }
            ]
            response = self._post(endpoint, close_payload)
            if response is None or (
                isinstance(response, list)
                and response
                and response[0].get("id") == ticket_id
            ):
                return True
            else:
                log.error(f"Final closure failed for {ticket_id}. Response: {response}")
                if (
                    isinstance(response, dict)
                    and "message" in response
                    and "Please assign this Ticket" in response["message"]
                ):
                    log.critical(
                        f"CRITICAL: Closure failed {ticket_id} due to 'assign agent' error. Agent in payload: {final_agent_id_for_closure}."
                    )
                return False
        except Exception as e:
            log.error(f"Exception during robust close {ticket_id}: {e}", exc_info=True)
            return False

    def create_ticket(
        self, ticket_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:  # ... (Same as before)
        if not self.customer_client_id:
            log.error("Cannot create: HALO_CUSTOMER_CLIENT_ID not set.")
            st.error("Create failed: App config missing.")
            return {"error": "App config missing."}
        ticket_data["client_id"] = self.customer_client_id
        ticket_data["use"] = "ticket"
        endpoint = f"/tickets?client_id={self.customer_client_id}"
        try:
            response_data = self._post(endpoint, [ticket_data])
            if isinstance(response_data, str):
                log.error(f"Ticket creation failed. API Response Text: {response_data}")
                try:
                    error_detail = json.loads(response_data)
                    return {"error": error_detail.get("message", response_data)}
                except json.JSONDecodeError:
                    return {"error": response_data}
            elif isinstance(response_data, dict) and "message" in response_data:
                log.error(
                    f"Ticket creation failed. API Error: {response_data['message']}"
                )
                log.error(f"API Error Details: {response_data.get('error', '')}")
                return {"error": response_data["message"]}
            elif (
                isinstance(response_data, list)
                and response_data
                and response_data[0].get("id") is not None
            ):
                return response_data[0]
            elif (
                isinstance(response_data, dict) and response_data.get("id") is not None
            ):
                return response_data
            else:
                log.error(
                    f"Ticket creation failed or unexpected response: {response_data}"
                )
                return {"error": f"Unexpected response: {response_data}"}
        except Exception as e:
            log.error(f"Error creating Halo ticket: {e}", exc_info=True)
            return {"error": str(e)}

    def search_users(
        self, query: str, max_results: int = 50
    ) -> list[User]:  # ... (Same as before)
        params = {
            "search": query,
            "includeinactive": "false",
            "page_size": max_results,
            "page_no": 1,
        }
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        raw = list(self._get_paged("/users", params, max_pages=1))
        return [User(**u) for u in raw]


# --- Streamlit Caching Functions (all same as before) ---
@st.cache_resource(ttl=300)
def search_users_cached(_halo_client: StandaloneHaloClient, query: str) -> list[User]:
    if not query or len(query.strip()) < 3:
        return []
    return _halo_client.search_users(query)


@st.cache_resource
def get_halo_client() -> StandaloneHaloClient:
    log.info("Initializing Halo Client for Streamlit session (using st.secrets).")
    try:
        client = StandaloneHaloClient()
        client._check_token()
        return client
    except Exception as e:
        st.error(f"Fatal: Failed Halo Client initialization: {e}", icon="🚨")
        log.exception("Fatal error during Halo client init.")
        st.stop()


@st.cache_resource(ttl=3600)
def get_statuses(_halo_client: StandaloneHaloClient) -> list[Status]:
    try:
        return _halo_client.get_statuses()
    except Exception as e:
        st.error(f"Error fetching statuses: {e}")
        return []


@st.cache_resource(ttl=3600)
def get_categories(_halo_client: StandaloneHaloClient) -> list[Category]:
    try:
        return _halo_client.get_categories()
    except Exception as e:
        st.error(f"Error fetching categories: {e}")
        return []


@st.cache_resource(ttl=3600)
def get_users(_halo_client: StandaloneHaloClient) -> list[User]:
    try:
        return _halo_client.get_users()
    except Exception as e:
        st.error(f"Error fetching users/agents: {e}")
        return []


@st.cache_resource(ttl=3600)
def get_teams(_halo_client: StandaloneHaloClient) -> list[dict]:
    try:
        return _halo_client.get_teams()
    except Exception as e:
        st.error(f"Error fetching teams: {e}")
        return []


@st.cache_resource(ttl=3600)
def get_sites(_halo_client: StandaloneHaloClient) -> list[Site]:
    try:
        return _halo_client.get_sites()
    except Exception as e:
        st.error(f"Error fetching sites: {e}")
        return []


@st.cache_resource(ttl=3600)
def get_ticket_types(_halo_client: StandaloneHaloClient) -> list[dict]:
    try:
        return _halo_client.get_ticket_types()
    except Exception as e:
        st.error(f"Error fetching ticket types: {e}")
        return []


@st.cache_resource(ttl=60, show_spinner="Fetching open tickets...")
def get_all_open_tickets_cached(
    _halo_client: StandaloneHaloClient, _cache_key=None
) -> list[Ticket]:
    try:
        return _halo_client.get_all_open_tickets()
    except Exception as e:
        st.error(f"Error fetching all open tickets: {e}", icon="⚠️")
        return []


@st.cache_resource(ttl=120, show_spinner="Searching tickets...")
def search_tickets_cached(
    _halo_client: StandaloneHaloClient, search_term: str, _cache_key=None
) -> list[Ticket]:
    log.info(
        f"Searching tickets (standard API /tickets?search=) via cache for term '{search_term}' (cache key: {_cache_key})..."
    )
    if not search_term or len(search_term.strip()) < 3:
        return []
    try:
        return _halo_client.search_tickets(search_term)
    except Exception as e:
        st.error(f"Error searching tickets for '{search_term}': {e}", icon="⚠️")
        log.exception(f"Error searching for '{search_term}'.")
        return []


@st.cache_resource(ttl=30, show_spinner=False)
def get_team_tickets_cached(
    _halo_client: StandaloneHaloClient, team_id: int, _cache_key=None
) -> list[Ticket]:
    try:
        return _halo_client.get_open_tickets_for_team(team_id)
    except Exception as e:
        st.error(f"Error fetching tickets for team ID {team_id}: {e}", icon="⚠️")
        log.exception(f"Error fetching tickets for team ID {team_id}.")
        return []


@st.cache_resource(ttl=300, show_spinner="Fetching ticket details...")
def get_ticket_details_cached(
    _halo_client: StandaloneHaloClient, ticket_id: int, _cache_key=None
) -> Optional[Ticket]:
    if not ticket_id:
        return None
    try:
        return _halo_client.get_ticket(ticket_id)
    except Exception as e:
        st.error(f"Error fetching ticket details {ticket_id}: {e}")
        return None


@st.cache_resource(ttl=60, show_spinner="Fetching ticket actions...")
def get_ticket_actions_cached(
    _halo_client: StandaloneHaloClient, ticket_id: int, _cache_key=None
) -> list[TicketAction]:
    if not ticket_id:
        return []
    try:
        return _halo_client.get_ticket_actions(ticket_id)
    except Exception as e:
        st.error(f"Error fetching ticket actions {ticket_id}: {e}")
        return []


def clear_ticket_cache(ticket_id: Optional[int] = None) -> None:  # ... (Same as before)
    st.cache_resource.clear()
    st.cache_data.clear()
    if ticket_id:
        st.session_state[f"ticket_details_key_{ticket_id}"] = time.time()
        st.session_state[f"ticket_actions_key_{ticket_id}"] = time.time()
    st.session_state["all_open_tickets_key"] = time.time()
    st.session_state["search_cache_key"] = time.time()
    if "team_data_cache_keys" not in st.session_state:
        st.session_state.team_data_cache_keys = {}
    teams_list = get_teams(get_halo_client())
    for team_detail in teams_list:
        if team_detail.get("id"):
            st.session_state.team_data_cache_keys[team_detail["id"]] = time.time()
    log.info(
        f"Cleared Streamlit cache (ticket_id: {ticket_id}) and refreshed team cache keys."
    )


def format_datetime(dt_str: Optional[str]) -> str:  # ... (Same as before)
    if not dt_str:
        return "N/A"
    try:
        if "+" not in dt_str and "Z" not in dt_str:
            dt_utc = datetime.fromisoformat(dt_str + "+00:00")
        else:
            dt_utc = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        dt_local = dt_utc.astimezone(TIMEZONE)
        return dt_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    except (ValueError, TypeError) as e:
        log.debug(f"Could not parse datetime '{dt_str}': {e}")
        return dt_str


def get_status_name(
    halo_client: StandaloneHaloClient, status_id: Optional[int]
) -> str:  # ... (Same as before)
    if status_id is None:
        return "N/A"
    statuses = get_statuses(halo_client)
    return next((s.name for s in statuses if s.id == status_id), f"ID:{status_id}")


def get_category_name(
    halo_client: StandaloneHaloClient, category_id: Optional[int]
) -> str:  # ... (Same as before)
    if category_id is None:
        return "N/A"
    categories = get_categories(halo_client)
    return next(
        (c.category_name for c in categories if c.id == category_id),
        f"ID:{category_id}",
    )


# --- UI Functions ---
def display_ticket_list(halo_client: StandaloneHaloClient):  # Reverted team tab logic
    st.title("Halo Ticket Interface v0.03")  # Removed (Dev Mode)
    if "active_search_query" not in st.session_state:
        st.session_state.active_search_query = None
    if "search_cache_key" not in st.session_state:
        st.session_state.search_cache_key = None
    if "search_query_text_input_val" not in st.session_state:
        st.session_state.search_query_text_input_val = ""

    with st.form(key="search_form"):
        search_cols = st.columns([0.7, 0.15, 0.15])
        with search_cols[0]:
            search_term_input = st.text_input(
                "Search tickets:",
                key="search_query_text_input_form",
                placeholder="Enter at least 3 characters...",
                value=st.session_state.search_query_text_input_val,
                label_visibility="collapsed",
            )
            st.session_state.search_query_text_input_val = search_term_input
        with search_cols[1]:
            search_submitted = st.form_submit_button(
                "🔍 Search", use_container_width=True
            )
        with search_cols[2]:
            clear_submitted = st.form_submit_button("Clear", use_container_width=True)
    if search_submitted:
        if search_term_input and len(search_term_input.strip()) >= 3:
            st.session_state.active_search_query = search_term_input.strip()
            st.session_state.search_cache_key = time.time()
            st.rerun()
        else:
            st.warning("Please enter at least 3 characters to search.")
    if clear_submitted:
        st.session_state.active_search_query = None
        st.session_state.search_cache_key = None
        st.session_state.search_query_text_input_val = ""
        st.rerun()
    st.write("---")
    list_cols_1 = st.columns([3, 1])
    with list_cols_1[0]:
        if st.button(
            "🔄 Refresh Display", key="refresh_list_top", use_container_width=True
        ):
            clear_ticket_cache()
            st.rerun()  # This will refresh all_open_tickets and team caches
    with list_cols_1[1]:
        if st.button(
            "➕ Create New Ticket", key="nav_create_ticket", use_container_width=True
        ):
            st.session_state.view = "create"
            st.session_state.current_ticket_id = None
            st.rerun()

    tickets_to_display = []
    display_header = ""
    if st.session_state.active_search_query:
        display_header = f"Search Results for: '{st.session_state.active_search_query}'"
        tickets_to_display = search_tickets_cached(
            halo_client,
            st.session_state.active_search_query,
            _cache_key=st.session_state.search_cache_key,
        )
    else:
        display_header = "Open Tickets by Team"
        open_tickets_cache_key = st.session_state.get(
            "all_open_tickets_key", time.time()
        )
        tickets_to_display = get_all_open_tickets_cached(
            halo_client, _cache_key=open_tickets_cache_key
        )
    st.header(display_header)
    status_map = {s.id: s.name for s in get_statuses(halo_client)}

    def format_ticket_for_search_summary(t: Ticket) -> dict:
        status_id_val = (
            t.status_id.id if t.status_id and t.status_id.id is not None else None
        )
        status_name = (
            status_map.get(status_id_val, f"ID:{status_id_val}")
            if status_id_val is not None
            else "N/A"
        )
        return {
            "ID": t.id,
            "Summary": t.summary or "N/A",
            "Status": status_name,
            "User": t.user_name or "N/A",
            "Site": t.site_name or "N/A",
        }

    if not tickets_to_display:
        if st.session_state.active_search_query:
            st.info("No tickets found matching your search criteria.")
        else:
            st.warning("No open tickets found or permission error. Check logs.")
    if tickets_to_display:
        if st.session_state.active_search_query:
            st.subheader(
                f"Top {min(MAX_SEARCH_RESULTS_TO_DISPLAY, len(tickets_to_display))} Search Results:"
            )
            for i, ticket in enumerate(
                tickets_to_display[:MAX_SEARCH_RESULTS_TO_DISPLAY]
            ):  # Show top N
                if ticket.id is None:
                    continue
                summary_data = format_ticket_for_search_summary(ticket)
                with st.container(border=True):
                    st.markdown(
                        f"**ID: {summary_data['ID']}** - {summary_data['Summary']}"
                    )
                    col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 1])
                    col_a.caption(f"Status: {summary_data['Status']}")
                    col_b.caption(f"User: {summary_data['User']}")
                    col_c.caption(f"Site: {summary_data['Site']}")
                    if col_d.button(
                        "Open", key=f"open_search_{ticket.id}", use_container_width=True
                    ):
                        st.session_state.view = "detail"
                        st.session_state.current_ticket_id = ticket.id
                        clear_ticket_cache(ticket.id)
                        st.rerun()
                st.write("---")
        else:  # Default tabbed view - REVERTED TO SIMPLER LOGIC
            category_map = {c.id: c.category_name for c in get_categories(halo_client)}

            def format_ticket_for_full_display(t: Ticket) -> dict:
                status_id_val = (
                    t.status_id.id
                    if t.status_id and t.status_id.id is not None
                    else None
                )
                category_id_val = (
                    t.categoryid_1.id
                    if t.categoryid_1 and t.categoryid_1.id is not None
                    else None
                )
                status_name = (
                    status_map.get(status_id_val, f"ID:{status_id_val}")
                    if status_id_val is not None
                    else "N/A"
                )
                category_name = (
                    category_map.get(category_id_val, f"ID:{category_id_val}")
                    if category_id_val is not None
                    else "N/A"
                )
                return {
                    "ID": t.id,
                    "Summary": t.summary or "N/A",
                    "Status": status_name,
                    "Category": category_name,
                    "User": t.user_name or "N/A",
                    "Site": t.site_name or "N/A",
                    "Team": t.team or "Unassigned",
                    "Last Update": format_datetime(t.last_update),
                }

            # Use the already fetched `tickets_to_display` for team bucketing
            tickets_by_team = defaultdict(list)
            active_teams_in_tickets = set()
            for t in tickets_to_display:  # Iterate over the "all open tickets"
                team_name = t.team if t.team else "Unassigned"
                tickets_by_team[team_name].append(t)
                if t.team:
                    active_teams_in_tickets.add(t.team)

            team_names_for_tabs = sorted(
                [name for name in tickets_by_team if name != "Unassigned"]
            )
            tab_titles = [f"All ({len(tickets_to_display)})"]
            if "Unassigned" in tickets_by_team:
                tab_titles.append(f"Unassigned ({len(tickets_by_team['Unassigned'])})")
            tab_titles.extend(
                [
                    f"{name} ({len(tickets_by_team[name])})"
                    for name in team_names_for_tabs
                ]
            )

            tabs = st.tabs(tab_titles)
            current_tab_index = 0
            with tabs[current_tab_index]:  # All tab
                df_all = pd.DataFrame(
                    [format_ticket_for_full_display(t) for t in tickets_to_display]
                )
                if "ID" in df_all.columns:
                    df_all = df_all.sort_values(by="ID", ascending=False)
                st.dataframe(
                    df_all, hide_index=True, use_container_width=True, height=600
                )
            current_tab_index += 1
            if "Unassigned" in tickets_by_team:
                with tabs[current_tab_index]:
                    df_unassigned = pd.DataFrame(
                        [
                            format_ticket_for_full_display(t)
                            for t in tickets_by_team["Unassigned"]
                        ]
                    )
                    if "ID" in df_unassigned.columns:
                        df_unassigned = df_unassigned.sort_values(
                            by="ID", ascending=False
                        )
                    st.dataframe(
                        df_unassigned,
                        hide_index=True,
                        use_container_width=True,
                        height=600,
                    )
                current_tab_index += 1
            for team_name in team_names_for_tabs:
                with tabs[current_tab_index]:
                    df_team = pd.DataFrame(
                        [
                            format_ticket_for_full_display(t)
                            for t in tickets_by_team[team_name]
                        ]
                    )
                    if "ID" in df_team.columns:
                        df_team = df_team.sort_values(by="ID", ascending=False)
                    st.dataframe(
                        df_team, hide_index=True, use_container_width=True, height=600
                    )
                current_tab_index += 1

            # Optional: Show inactive teams still (less critical now)
            teams_from_api = get_teams(halo_client)  # Fetches active teams
            all_known_team_names = {t["name"] for t in teams_from_api if t.get("name")}
            teams_without_open_tickets_in_view = (
                all_known_team_names - active_teams_in_tickets - {"Unassigned"}
            )
            if teams_without_open_tickets_in_view:
                st.caption(
                    f"Note: The following active teams have no open tickets in this view: {', '.join(sorted(list(teams_without_open_tickets_in_view)))}"
                )

    st.write("---")
    st.subheader("Open Ticket by ID")
    with st.form(key="open_ticket_id_form"):
        open_ticket_cols_form = st.columns([0.85, 0.15])
        with open_ticket_cols_form[0]:
            selected_id_input_form = st.number_input(
                "Ticket ID:",
                min_value=1,
                step=1,
                value=None,
                key="open_ticket_input_main_form",
                label_visibility="collapsed",
                placeholder="Enter Ticket ID...",
            )
        with open_ticket_cols_form[1]:
            open_ticket_submitted = st.form_submit_button(
                "Open", use_container_width=True
            )
    if open_ticket_submitted and selected_id_input_form is not None:
        st.session_state.view = "detail"
        st.session_state.current_ticket_id = int(selected_id_input_form)
        clear_ticket_cache(int(selected_id_input_form))
        st.rerun()


# display_ticket_detail and add_new_note are the same as your last fully working version with inline image upload attempts
def display_ticket_detail(
    halo_client: StandaloneHaloClient, ticket_id: int
):  # Same logic as your previous complete version
    st.title(f"Halo Ticket #{ticket_id}")
    details_cache_key = st.session_state.get(
        f"ticket_details_key_{ticket_id}", time.time()
    )
    actions_cache_key = st.session_state.get(
        f"ticket_actions_key_{ticket_id}", time.time()
    )
    ticket = get_ticket_details_cached(
        halo_client, ticket_id, _cache_key=details_cache_key
    )
    main_cols = st.columns([3, 1])
    with main_cols[0]:
        if not ticket:
            st.error(f"Could not load details for ticket {ticket_id}.", icon="❌")
            return
        st.subheader("Ticket Details")
        info_cols = st.columns(3)
        current_status_id = ticket.status_id.id if ticket and ticket.status_id else None
        info_cols[0].metric("Status", get_status_name(halo_client, current_status_id))
        info_cols[1].metric("User", ticket.user_name or "N/A")
        info_cols[2].metric("Site", ticket.site_name or "N/A")
        st.text_area(
            "Summary",
            value=ticket.summary or "",
            disabled=True,
            key=f"summary_{ticket_id}",
            height=100,
        )
        st.subheader("Add New Note / Email User")
        note_key = f"new_note_text_{ticket_id}"
        if note_key not in st.session_state:
            st.session_state[note_key] = ""
        note_text = st.text_area(
            "Note / Email Content:",
            value=st.session_state[note_key],
            key=f"text_area_{note_key}",
            height=150,
        )
        uploaded_files = st.file_uploader(
            "Attach Images (will be appended to note)",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg", "gif"],
            key=f"file_uploader_{ticket_id}",
        )
        note_action_cols = st.columns(2)
        with note_action_cols[0]:
            if st.button(
                "Add Private Note",
                key=f"add_private_note_{ticket_id}",
                use_container_width=True,
            ):
                if note_text.strip() or uploaded_files:
                    with st.spinner("Adding note..."):
                        if add_new_note(
                            halo_client,
                            ticket_id,
                            note_text,
                            uploaded_files,
                            is_private=True,
                        ):
                            st.session_state[note_key] = ""
                            st.rerun()
                else:
                    st.warning("Enter note content or upload an image.")
        with note_action_cols[1]:
            if st.button(
                "✉️ Add Note & Email User",
                key=f"email_user_{ticket_id}",
                use_container_width=True,
            ):
                if note_text.strip() or uploaded_files:
                    with st.spinner("Sending..."):
                        if add_new_note(
                            halo_client,
                            ticket_id,
                            note_text,
                            uploaded_files,
                            is_private=False,
                            send_email=True,
                        ):
                            st.session_state[note_key] = ""
                            st.rerun()
                else:
                    st.warning("Enter email content or upload an image.")
        st.write("---")
        st.subheader("Classification & Assignment")
        field_cols = st.columns(4)
        with field_cols[0]:
            statuses = get_statuses(halo_client)
            status_options = {s.id: s.name for s in statuses}
            if (
                current_status_id not in status_options
                and current_status_id is not None
            ):
                status_options[current_status_id] = (
                    f"Current ({get_status_name(halo_client, current_status_id)})"
                )
            selected_status_id = st.selectbox(
                "Status",
                options=list(status_options.keys()),
                format_func=lambda x: status_options.get(x, f"ID:{x}"),
                index=list(status_options.keys()).index(current_status_id)
                if current_status_id in status_options
                else 0,
                key=f"status_select_{ticket_id}",
                label_visibility="collapsed",
            )
            if selected_status_id != current_status_id:
                if st.button(
                    "Apply Status",
                    key=f"apply_status_{ticket_id}",
                    use_container_width=True,
                ):
                    with st.spinner("Updating..."):
                        if halo_client.update_ticket_field(
                            ticket_id, "status_id", selected_status_id
                        ):
                            clear_ticket_cache(ticket_id)
                            st.rerun()
                        else:
                            st.error("Update failed.")
        with field_cols[1]:
            categories = get_categories(halo_client)
            category_options = {c.id: c.category_name for c in categories}
            current_category_id = ticket.categoryid_1.id if ticket.categoryid_1 else 0
            if current_category_id != 0 and current_category_id not in category_options:
                category_options[current_category_id] = (
                    f"Current ({get_category_name(halo_client, current_category_id)})"
                )
            category_options_with_none = {0: "--- None ---", **category_options}
            selected_category_id = st.selectbox(
                "Category",
                options=list(category_options_with_none.keys()),
                format_func=lambda x: category_options_with_none.get(x, f"ID:{x}"),
                index=list(category_options_with_none.keys()).index(
                    current_category_id
                ),
                key=f"category_select_{ticket_id}",
                label_visibility="collapsed",
            )
            if selected_category_id != current_category_id:
                if st.button(
                    "Apply Category",
                    key=f"apply_category_{ticket_id}",
                    use_container_width=True,
                ):
                    with st.spinner("Updating..."):
                        api_cat_value = (
                            selected_category_id if selected_category_id != 0 else None
                        )
                        if halo_client.update_ticket_field(
                            ticket_id, "categoryid_1", api_cat_value
                        ):
                            clear_ticket_cache(ticket_id)
                            st.rerun()
                        else:
                            st.error("Update failed.")
        with field_cols[2]:
            agents = get_users(halo_client)
            agent_options = {a.id: a.name for a in agents if a.name}
            current_agent_id = ticket.agent_id or 0
            if current_agent_id != 0 and current_agent_id not in agent_options:
                agent_name = next(
                    (
                        u.name
                        for u in get_users(halo_client)
                        if u.id == current_agent_id
                    ),
                    f"ID:{current_agent_id}",
                )
                agent_options[current_agent_id] = (
                    agent_name if agent_name else f"ID:{current_agent_id}"
                )
            agent_options_with_none = {0: "--- Unassigned ---", **agent_options}
            selected_agent_id = st.selectbox(
                "Agent",
                options=list(agent_options_with_none.keys()),
                format_func=lambda x: agent_options_with_none.get(x, f"ID:{x}"),
                index=list(agent_options_with_none.keys()).index(current_agent_id),
                key=f"agent_select_{ticket_id}",
                label_visibility="collapsed",
            )
            if selected_agent_id != current_agent_id:
                if st.button(
                    "Apply Agent",
                    key=f"apply_agent_{ticket_id}",
                    use_container_width=True,
                ):
                    with st.spinner("Updating..."):
                        if halo_client.update_ticket_field(
                            ticket_id, "agent_id", selected_agent_id
                        ):
                            clear_ticket_cache(ticket_id)
                            st.rerun()
                        else:
                            st.error("Update failed.")
        with field_cols[3]:
            teams = get_teams(halo_client)
            team_options = {
                t["name"]: t["name"]
                for t in teams
                if t.get("name") and not t.get("inactive")
            }
            current_team_name = ticket.team or ""
            if current_team_name and current_team_name not in team_options:
                team_options[current_team_name] = f"Current ({current_team_name})"
            team_options_with_none = {"": "--- Unassigned ---", **team_options}
            selected_team_name = st.selectbox(
                "Team",
                options=list(team_options_with_none.keys()),
                format_func=lambda x: team_options_with_none.get(x, x),
                index=list(team_options_with_none.keys()).index(current_team_name),
                key=f"team_select_{ticket_id}",
                label_visibility="collapsed",
            )
            if selected_team_name != current_team_name:
                if st.button(
                    "Apply Team",
                    key=f"apply_team_{ticket_id}",
                    use_container_width=True,
                ):
                    with st.spinner("Updating..."):
                        if halo_client.update_ticket_field(
                            ticket_id, "team", selected_team_name or None
                        ):
                            clear_ticket_cache(ticket_id)
                            st.rerun()
                        else:
                            st.error("Update failed.")
        st.subheader("Details")
        details_container = st.container(border=True)
        with details_container:
            if ticket.details_html:
                safe_html = ticket.details_html.replace("<script", "<script").replace(
                    "</script>", "</script>"
                )
                st.markdown(safe_html, unsafe_allow_html=True)
            elif ticket.details:
                st.text(ticket.details)
            else:
                st.markdown("_No details provided._")
        st.subheader("History / Notes")
        actions = get_ticket_actions_cached(
            halo_client, ticket_id, _cache_key=actions_cache_key
        )
        if not actions:
            st.info("No actions or notes found.")
        else:
            actions.sort(key=lambda a: a.datetime or "1970", reverse=True)
            for action in actions:
                creator = action.createdby or action.who or "System"
                exp_title = f"{format_datetime(action.datetime)} - {action.outcome or 'Action'} by {creator} (ID: {action.id})"
                with st.expander(exp_title, expanded=False):
                    header_lines = []
                    if action.emailfrom:
                        header_lines.append(f"**From:** {action.emailfrom}")
                    if action.emailto:
                        header_lines.append(f"**To:** {action.emailto}")
                    if action.emailcc:
                        header_lines.append(f"**Cc:** {action.emailcc}")
                    if action.subject:
                        header_lines.append(f"**Subject:** {action.subject}")
                    if header_lines:
                        st.markdown("<br>".join(header_lines), unsafe_allow_html=True)
                        st.markdown("---")
                    if action.note_html:
                        safe_note_html = action.note_html.replace(
                            "<script", "<script"
                        ).replace("</script>", "</script>")
                        st.markdown(safe_note_html, unsafe_allow_html=True)
                    elif action.note:
                        st.text(action.note)
                    else:
                        st.markdown("_No note content._")
    with main_cols[1]:
        st.subheader("Actions")
        if st.button(
            "⬅️ Back to List", key="back_button_detail_sidebar", use_container_width=True
        ):
            st.session_state.view = "list"
            st.session_state.current_ticket_id = None
            st.rerun()
        if st.button(
            "🔄 Refresh Details",
            key="refresh_detail_button_sidebar",
            use_container_width=True,
        ):
            clear_ticket_cache(ticket_id)
            st.rerun()
        st.write("---")
        if ticket and current_status_id != TICKET_STATUS_CLOSED:
            if st.button(
                "🔒 Close Ticket",
                key="close_ticket_button_detail_sidebar",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Closing ticket..."):
                    ok = halo_client.close_ticket_with_agent(
                        ticket_id, agent_id=DEFAULT_AGENT_ID
                    )
                if ok:
                    st.success("Ticket closed.")
                    clear_ticket_cache(ticket_id)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to close ticket.")
        elif ticket:
            st.info("Ticket is already closed.")


def add_new_note(
    halo_client: StandaloneHaloClient,
    ticket_id: int,
    note_content: str,
    uploaded_files: Optional[List[Any]],
    is_private: bool = True,
    send_email: bool = False,
) -> bool:  # Same as before, uses corrected upload
    outcome = (
        "Private Note" if is_private else ("Email Sent" if send_email else "Note Added")
    )
    final_note_content = note_content if note_content else ""
    final_note_html = (
        f"<p>{final_note_content.replace(chr(10), '<br>')}</p>"
        if final_note_content
        else "<p></p>"
    )
    image_markdown_links = []
    image_html_tags = []
    if uploaded_files:
        all_images_uploaded_successfully = True
        with st.spinner("Uploading images..."):
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                file_name = uploaded_file.name
                image_url_token = halo_client.upload_attachment_and_get_token(
                    ticket_id, file_name, file_bytes
                )
                if image_url_token:
                    image_markdown_links.append(f"![{file_name}]({image_url_token})")
                    image_html_tags.append(
                        f'<img src="{image_url_token}" alt="{file_name}" />'
                    )
                    log.info(
                        f"Image '{file_name}' uploaded, token path: {image_url_token}"
                    )
                else:
                    st.error(f"Failed to upload image: {file_name}")
                    log.error(
                        f"Failed to upload image {file_name} for ticket {ticket_id}"
                    )
                    all_images_uploaded_successfully = False
        if not all_images_uploaded_successfully:
            st.warning(
                "Some images failed to upload. Note will be added with successfully uploaded images only."
            )
        if image_markdown_links:
            final_note_content += "\n\n--- Attached Images ---\n" + "\n".join(
                image_markdown_links
            )
        if image_html_tags:
            final_note_html += (
                "<hr/><p><strong>Attached Images:</strong></p>"
                + "".join(image_html_tags)
            )
    if not final_note_content.strip() and not image_markdown_links:
        return False
    payload = {
        "ticket_id": ticket_id,
        "outcome": outcome,
        "note": final_note_content,
        "note_html": final_note_html,
        "hiddenfromuser": is_private,
        "sendemail": send_email,
    }
    if send_email:
        details_cache_key = st.session_state.get(
            f"ticket_details_key_{ticket_id}", time.time()
        )
        ticket = get_ticket_details_cached(
            halo_client, ticket_id, _cache_key=details_cache_key
        )
        if ticket and ticket.user_email:
            payload["emailto"] = ticket.user_email
        else:
            log.error(
                f"Cannot send email for ticket {ticket_id}: User email not found."
            )
            st.error("Cannot send email: Ticket contact email not found.")
            return False
    if halo_client.add_ticket_action(payload):
        st.success(f"Action '{outcome}' added.")
        clear_ticket_cache(ticket_id)
        return True
    else:
        st.error(f"Failed to add action '{outcome}'.")
        return False


def display_create_ticket_form(
    halo_client: StandaloneHaloClient,
):  # ... (Same as before)
    st.title("Create New Halo Ticket")
    if not HALO_CUSTOMER_CLIENT_ID:
        st.error(
            "Ticket creation unavailable: `HALO_CUSTOMER_CLIENT_ID` not configured.",
            icon="🚫",
        )
        return
    if st.button("⬅️ Back to List", key="back_button_create"):
        st.session_state.view = "list"
        st.session_state.current_ticket_id = None
        st.rerun()
    st.write("---")
    with st.spinner("Loading form data..."):
        statuses = get_statuses(halo_client)
        categories = get_categories(halo_client)
        sites = get_sites(halo_client)
        teams = [
            t for t in get_teams(halo_client) if t.get("name") and not t.get("inactive")
        ]
        types = get_ticket_types(halo_client)
        agents = [u for u in get_users(halo_client) if not u.inactive and u.name]
    site_opts = {s.id: s.name for s in sites}
    status_opts = {s.id: s.name for s in statuses}
    cat_opts = {c.id: c.category_name for c in categories}
    agent_opts = {a.id: a.name for a in agents}
    team_opts = {t["name"]: t["name"] for t in teams if t.get("name")}
    type_opts = {t["id"]: t["name"] for t in types if t.get("id") and t.get("name")}
    if "create_form_selected_user" not in st.session_state:
        st.session_state.create_form_selected_user = None
    with st.form("create_ticket_form"):
        st.subheader("Ticket Information")
        summary = st.text_input("Summary*", help="Brief description")
        details = st.text_area("Details*", help="Full description", height=200)
        st.subheader("Reporter")
        user_search_query = st.text_input(
            "Find user by name or email*", placeholder="Start typing..."
        )
        selected_user_display = "No user selected."
        if st.session_state.create_form_selected_user:
            sel_usr = st.session_state.create_form_selected_user
            selected_user_display = (
                f"Selected: {sel_usr.name} ({sel_usr.emailaddress or 'No email'})"
            )
        st.caption(selected_user_display)
        if len(user_search_query.strip()) >= 3:
            user_matches = search_users_cached(halo_client, user_search_query)
            if user_matches:
                user_labels = {
                    u.id: f"{u.name} – {u.emailaddress or 'no-email'}"
                    for u in user_matches
                }
                picked_user_id = st.selectbox(
                    "Select matching user*",
                    list(user_labels.keys()),
                    format_func=lambda i: user_labels[i],
                    index=None,
                    placeholder="Select from matches...",
                )
                if picked_user_id:
                    sel_user_obj = next(
                        (u for u in user_matches if u.id == picked_user_id), None
                    )
                    if sel_user_obj:
                        st.session_state.create_form_selected_user = sel_user_obj
            elif user_search_query:
                st.warning("No users found matching your search.")
        st.subheader("Location & Classification")
        create_cols1 = st.columns(4)
        with create_cols1[0]:
            site_id = st.selectbox(
                "Site*",
                site_opts.keys(),
                format_func=site_opts.get,
                index=None,
                placeholder="Select Site...",
            )
        with create_cols1[1]:
            type_id = st.selectbox(
                "Ticket Type*",
                type_opts.keys(),
                format_func=type_opts.get,
                index=list(type_opts.keys()).index(DEFAULT_TICKET_TYPE_ID)
                if DEFAULT_TICKET_TYPE_ID in type_opts
                else 0,
            )
        with create_cols1[2]:
            status_id = st.selectbox(
                "Initial Status*",
                status_opts.keys(),
                format_func=status_opts.get,
                index=list(status_opts.keys()).index(DEFAULT_NEW_STATUS_ID)
                if DEFAULT_NEW_STATUS_ID in status_opts
                else 0,
            )
        with create_cols1[3]:
            cat_id = st.selectbox(
                "Category",
                [0] + list(cat_opts.keys()),
                format_func=lambda x: "--- None ---"
                if x == 0
                else cat_opts.get(x, f"ID:{x}"),
            )
        st.subheader("Priority & Assignment (Optional)")
        create_cols2 = st.columns(4)
        with create_cols2[0]:
            impact_id = st.selectbox(
                "Impact",
                [1, 2, 3, 4],
                format_func=lambda x: {1: "Crit", 2: "High", 3: "Med", 4: "Low"}[x],
                index=2,
            )
        with create_cols2[1]:
            urgency_id = st.selectbox(
                "Urgency",
                [1, 2, 3, 4],
                format_func=lambda x: {1: "Crit", 2: "High", 3: "Med", 4: "Low"}[x],
                index=1,
            )
        with create_cols2[2]:
            agent_id = st.selectbox(
                "Assign Agent",
                [0] + list(agent_opts.keys()),
                format_func=lambda x: "--- Unassigned ---"
                if x == 0
                else agent_opts.get(x, f"ID:{x}"),
            )
        with create_cols2[3]:
            team_name = st.selectbox(
                "Assign Team",
                [""] + list(team_opts.keys()),
                format_func=lambda x: "--- Unassigned ---"
                if x == ""
                else team_opts.get(x, x),
            )
        st.write("---")
        submitted = st.form_submit_button(
            "Create Ticket", use_container_width=True, type="primary"
        )
    if submitted:
        missing = []
        if not summary:
            missing.append("Summary")
        if not details:
            missing.append("Details")
        if not st.session_state.create_form_selected_user:
            missing.append("Reporter (Search and select a user)")
        if not site_id:
            missing.append("Site")
        if missing:
            st.error("Please fill in required fields: " + ", ".join(missing))
        else:
            sel_user = st.session_state.create_form_selected_user
            payload = {
                "summary": summary,
                "details": details,
                "user_id": sel_user.id,
                "user_email": sel_user.emailaddress,
                "site_id": site_id,
                "status_id": status_id,
                "tickettype_id": type_id,
                "reportedby": sel_user.emailaddress,
                "dateoccurred": datetime.now(pytz.utc).isoformat(),
                "impact": impact_id,
                "urgency": urgency_id,
            }
            if cat_id != 0:
                payload["categoryid_1"] = cat_id
            if agent_id != 0:
                payload["agent_id"] = agent_id
            if team_name:
                payload["team"] = team_name
            with st.spinner("Creating ticket..."):
                result = halo_client.create_ticket(payload)
            if result and result.get("id"):
                new_id = result["id"]
                st.success(f"Ticket #{new_id} created!")
                st.session_state.create_form_selected_user = None
                clear_ticket_cache()
                st.session_state.view = "detail"
                st.session_state.current_ticket_id = new_id
                st.rerun()
            else:
                error_msg = result.get("error", "Unknown error.")
                st.error(f"Creation failed: {error_msg}")
                log.error(
                    f"Ticket creation failed. Payload: {payload}, Response: {result}"
                )


# --- Main App Execution ---
if "view" not in st.session_state:
    st.session_state.view = "list"
if "current_ticket_id" not in st.session_state:
    st.session_state.current_ticket_id = None

halo_client_instance = get_halo_client()

if halo_client_instance:
    if st.session_state.view == "list":
        display_ticket_list(halo_client_instance)
    elif st.session_state.view == "create":
        display_create_ticket_form(halo_client_instance)
    elif st.session_state.view == "detail" and st.session_state.current_ticket_id:
        display_ticket_detail(halo_client_instance, st.session_state.current_ticket_id)
    else:
        st.session_state.view = "list"
        st.session_state.current_ticket_id = None
        st.rerun()
