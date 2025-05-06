# halo_interface.py
# COMPLETE FILE - VERSION 10 - Removed client_id filter for all tickets, added inactive teams
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging
import time
import os
import json
from typing import Any, Dict, Optional, List, Generator, Callable
import httpx
from pydantic import BaseModel, Field, ValidationError
import dotenv
from collections import defaultdict

# --- Configuration & Setup ---
st.set_page_config(layout="wide", page_title="Halo Ticket Interface")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Configure httpx logging to see requests/responses
# logging.getLogger("httpx").setLevel(logging.DEBUG) # Uncomment for very detailed HTTP logs

log = logging.getLogger(__name__)

try:
    TIMEZONE = pytz.timezone("Australia/Perth")
except pytz.UnknownTimeZoneError:
    log.warning("Timezone 'Australia/Perth' not found. Using UTC.")
    TIMEZONE = pytz.utc

TICKET_STATUS_CLOSED = 9
DEFAULT_NEW_STATUS_ID = 1
DEFAULT_AGENT_ID = 0
DEFAULT_TEAM_NAME = ""
DEFAULT_IMPACT = 3
DEFAULT_URGENCY = 2
DEFAULT_TICKET_TYPE_ID = 1


# --- Inlined Models ---
class HaloBaseModel(BaseModel):
    class Config:
        extra = "ignore"

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        if hasattr(super(), "model_dump"):
            return super().model_dump(*args, **kwargs)
        else:
            return super().dict(*args, **kwargs)  # type: ignore

    def model_dump_json(self, *args, **kwargs) -> str:
        kwargs.setdefault("exclude_none", True)
        if hasattr(super(), "model_dump_json"):
            return super().model_dump_json(*args, **kwargs)
        else:
            return super().json(*args, **kwargs)  # type: ignore


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
    # Add other fields if needed later


class Site(HaloBaseModel):
    id: int
    name: str
    client_id: int | None = None
    client_name: str | None = None
    inactive: bool = False
    # Add other fields if needed later


class Category(HaloBaseModel):
    id: int
    category_name: str
    # Add other fields if needed later


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
    # Simplified for list view, add others if needed
    details: str | None = None  # Keep details for specific fetch
    details_html: str | None = None  # Keep details for specific fetch
    customfields: Optional[List[Dict[str, Any]]] = (
        None  # Keep details for specific fetch
    )


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
    # Add other fields if needed later


class Status(HaloBaseModel):
    id: int
    name: str


# --- Inlined Utilities ---
def get_env_var(name: str, var_type: type, required=True, default=None):
    value = os.environ.get(name)
    if required and value is None:
        raise ValueError(f"Missing required env var: '{name}'")
    if value is None:
        return default
    try:
        return var_type(value)
    except ValueError:
        raise ValueError(
            f"Env var '{name}' ('{value}') cannot be cast to {var_type.__name__}."
        )


# --- Inlined HaloClient ---
class StandaloneHaloClient:
    def __init__(self) -> None:
        dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())
        self.secret: str = get_env_var("HALO_SECRET", str)
        self.client_id: str = get_env_var("HALO_CLIENT_ID", str)
        # Store customer_client_id for specific actions, but don't use it for general ticket list
        self.customer_client_id: Optional[int] = get_env_var(
            "HALO_CUSTOMER_CLIENT_ID", int, required=False
        )
        if self.customer_client_id:
            log.info(
                f"Customer Client ID set to: {self.customer_client_id}. Will be used for specific actions."
            )
        else:
            log.warning(
                "HALO_CUSTOMER_CLIENT_ID not set in env. Some actions (create, update) might require it."
            )

        self.tenant: str = get_env_var("HALO_TENANT", str)
        self.api_host_url: str = f"https://{self.tenant}.haloitsm.com"
        self.api_auth_url: str = f"{self.api_host_url}/auth/token"
        self.api_url: str = f"{self.api_host_url}/api"
        self._client: httpx.Client = httpx.Client(
            headers={"Accept": "application/json"}, timeout=60.0
        )
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        log.info(f"Halo Client configured for tenant: {self.tenant}")

    def _auth(self, grant_type="client_credentials"):
        log.info(f"Attempting Halo authentication (grant_type: {grant_type})...")
        body = {
            "grant_type": grant_type,
            "client_id": self.client_id,
            "client_secret": self.secret,
            "scope": ["all"],
        }
        # Removed password auth logic for simplicity, assuming client_credentials
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
            log.info(
                f"Halo authentication successful. Token expires around: %s",
                self._token_expiry.astimezone(TIMEZONE),
            )
        except Exception as e:
            log.error(f"Halo auth error: {e}", exc_info=True)
            raise ConnectionError(f"Halo Auth Failed: {e}") from e

    def _check_token(self):
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
    ) -> httpx.Response:
        self._check_token()
        url = f"{self.api_url}{endpoint}"
        headers = self._client.headers.copy()
        try:
            # Log the actual request being sent
            log.info(
                f"Requesting: {method} {url} | Params: {params} | JSON Body: {'Present' if json_data else 'None'}"
            )
            response = self._client.request(
                method, url, params=params, json=json_data, headers=headers
            )
            log.info(
                f"Response: {response.status_code} {response.reason_phrase} from {url}"
            )
            log.debug(
                f"Response Body Snippet: {response.text[:500]}..."
            )  # Debug level for body

            if response.status_code == 401:
                log.warning("Retrying after 401 Unauthorized...")
                self._access_token = None
                self._token_expiry = None
                self._auth()
                headers["Authorization"] = f"Bearer {self._access_token}"
                log.info(f"Retrying Request: {method} {url} | Params: {params}")
                response = self._client.request(
                    method, url, params=params, json=json_data, headers=headers
                )
                log.info(
                    f"Retry Response: {response.status_code} {response.reason_phrase} from {url}"
                )

            # Raise HTTP errors (4xx, 5xx) for easier debugging, except 400 on POST /tickets
            is_ticket_post = method.upper() == "POST" and endpoint.startswith(
                "/tickets"
            )
            if not is_ticket_post or response.status_code != 400:
                response.raise_for_status()

            return response
        except httpx.RequestError as e:
            log.error(
                f"Network error during Halo API request to {url}: {e}", exc_info=True
            )
            raise ConnectionError(f"Network error connecting to Halo API: {e}") from e
        except httpx.HTTPStatusError as e:
            log.error(
                f"HTTP error from Halo API ({e.response.status_code}) for {url}: {e.response.text}",
                exc_info=True,
            )
            # Specific error types for common issues
            if e.response.status_code == 404:
                raise FileNotFoundError(f"Resource not found at {url}") from e
            if e.response.status_code == 403:
                raise PermissionError(
                    f"Permission denied for {url}. Check API key scope."
                ) from e
            raise ConnectionError(
                f"HTTP error {e.response.status_code} from Halo API: {e.response.text}"
            ) from e
        except Exception as e:
            log.error(
                f"Unexpected error during Halo API request to {url}: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed Halo API request: {e}") from e

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        response = self._request("GET", endpoint, params=params)
        try:
            return response.json()
        except json.JSONDecodeError:
            log.error(f"Bad JSON from GET {endpoint}: {response.text}")
            raise ValueError(f"Bad JSON received from Halo API GET {endpoint}")

    def _post(self, endpoint: str, data: Any) -> Any:
        response = self._request("POST", endpoint, json_data=data)
        is_ticket_create = endpoint.startswith("/tickets")
        if is_ticket_create and response.status_code == 400:
            log.warning(
                f"POST {endpoint} received 400 Bad Request. Response: {response.text}"
            )
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text  # Return raw text if not JSON

        try:
            if response.status_code == 204:
                return None  # No Content success
            return response.json()
        except json.JSONDecodeError:
            log.error(
                f"Bad JSON from POST {endpoint} (Status: {response.status_code}): {response.text}"
            )
            return {
                "error": f"Bad JSON response (Status: {response.status_code})",
                "response_text": response.text,
            }

    def _get_paged(
        self, endpoint: str, params: Optional[Dict] = None, max_pages=100
    ) -> Generator[Dict, Any, None]:
        if params is None:
            params = {}
        page_size = params.get("page_size", 100)
        params["page_size"] = page_size
        params["page_no"] = 1
        pages_fetched = 0
        total_items_yielded = 0
        log.info(f"Starting paged fetch for {endpoint}...")  # Params logged in _request

        while pages_fetched < max_pages:
            log.debug(
                f"Fetching page {params['page_no']} for {endpoint} (Page Size: {page_size})..."
            )
            pages_fetched += 1
            try:
                data = self._get(endpoint, params)
                items = []
                record_count = 0  # API's reported total count for the query
                items_on_page = 0

                if isinstance(data, list):
                    items = data
                    record_count = len(
                        items
                    )  # Simple list, count is length (may not be total if paged by API internally?)
                    items_on_page = len(items)
                    log.debug(
                        f"Page {params['page_no']}: Received direct list response with {items_on_page} items."
                    )
                elif isinstance(data, dict):
                    record_count = data.get("record_count", 0)
                    # Try to find the list: common pattern is key matching endpoint name (plural)
                    list_key = endpoint.split("/")[-1].lower()
                    if list_key in data and isinstance(data[list_key], list):
                        items = data[list_key]
                    else:  # Fallback: find first list value
                        key = next(
                            (k for k, v in data.items() if isinstance(v, list)), None
                        )
                        items = data.get(key, []) if key else []
                    items_on_page = len(items)
                    log.debug(
                        f"Page {params['page_no']}: Received dict response. API reports {record_count} total records. Found {items_on_page} items in list key '{list_key or key}'."
                    )
                else:
                    log.warning(
                        f"Unexpected data type {type(data)} received for {endpoint} page {params['page_no']}. Stopping."
                    )
                    break

                if not items:
                    log.info(
                        f"No more items found on page {params['page_no']} for {endpoint}. API reported {record_count} total records. Total yielded so far: {total_items_yielded}."
                    )
                    break

                yield from items
                total_items_yielded += items_on_page

                # Check if we should continue pagination
                # 1. If API provides record_count and we've yielded that many or more
                if record_count > 0 and total_items_yielded >= record_count:
                    log.info(
                        f"Yielded {total_items_yielded} items, >= API reported record_count {record_count}. Reached expected end for {endpoint}."
                    )
                    break
                # 2. If the number of items returned is less than the requested page size (heuristic)
                elif items_on_page < page_size:
                    log.info(
                        f"Fetched {items_on_page} items (less than page size {page_size}). Assuming last page for {endpoint}. Total yielded: {total_items_yielded}."
                    )
                    break

                # Prepare for next page
                params["page_no"] += 1
                time.sleep(0.2)  # Be nice to the API

            except Exception as e:
                log.error(
                    f"Error fetching page {params['page_no']} for {endpoint}: {e}",
                    exc_info=True,
                )
                break  # Stop paging on error

        log.info(
            f"Finished paged fetch for {endpoint} after {pages_fetched} page(s). Total items yielded: {total_items_yielded}."
        )
        if pages_fetched >= max_pages:
            log.warning(
                f"Reached maximum page limit ({max_pages}) fetching from {endpoint}. Data might be incomplete."
            )

    def _get_all_paged(
        self, endpoint: str, params: Optional[Dict] = None, max_pages=100
    ) -> List[Dict]:
        return list(self._get_paged(endpoint, params, max_pages=max_pages))

    # --- Core Data Fetching Methods ---

    def get_statuses(self) -> list[Status]:
        # Statuses are usually global, no client filter needed
        data = self._get("/status")
        items = data.get("status", []) if isinstance(data, dict) else data
        return [Status(**s) for s in items]

    def get_categories(self) -> list[Category]:
        # Categories might be client-specific in some setups, keep client_id if needed
        params = {}
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        items = self._get_all_paged("/category", params)
        return [Category(**c) for c in items if c]

    def get_users(self) -> list[User]:
        # Users are often associated with a client, keep filter if needed
        params = {"includeinactive": "false"}
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        items = self._get_all_paged("/users", params)
        return [User(**u) for u in items if u]

    def get_teams(self) -> list[dict]:
        # Fetch ALL teams, including inactive, no client filter by default
        params = {"includeinactive": "true"}
        log.info(f"Fetching teams with params: {params}")  # Log params used
        return self._get_all_paged("/team", params)

    def get_sites(self) -> list[Site]:
        # Sites are usually linked to clients
        params = {}
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        items = self._get_all_paged("/site", params)
        return [Site(**s) for s in items if s]

    def get_ticket_types(self) -> list[dict]:
        # Ticket Types are often global
        return self._get_all_paged("/tickettype")

    def get_user_by_email(self, email: str) -> Optional[User]:
        if not email:
            return None
        params = {"emailaddress": email, "includeinactive": "true"}
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id
        users_data = list(self._get_paged("/users", params, max_pages=1))
        # ... (rest of parsing logic remains the same) ...
        if users_data:
            try:
                for user_data in users_data:
                    if user_data.get("emailaddress", "").lower() == email.lower():
                        return User(**user_data)
                log.warning(
                    f"API search returned user(s) for '{email}', but none matched exactly."
                )
                return None
            except Exception as e:
                log.error(f"Error parsing user data for email {email}: {e}")
                return None
        log.info(f"No user found with email: {email}")
        return None

    def get_ticket(self, ticket_id: int) -> Optional[Ticket]:
        if not ticket_id:
            return None
        # Use customer_client_id here if available, might be necessary for permissions
        params = {
            "includedetails": True,
            "includeCustomFields": True,
            "includeHtmlDetails": True,
        }
        if self.customer_client_id:
            params["client_id"] = self.customer_client_id

        try:
            data = self._get(f"/tickets/{ticket_id}", params)
            # ... (rest of parsing logic remains the same) ...
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

    # --- UPDATED METHOD TO FETCH ALL OPEN TICKETS (No client_id) ---
    def get_all_open_tickets(self, max_pages: int = 100) -> list[Ticket]:
        """Fetches ALL open tickets using pagination, without client_id filter."""
        params = {
            # 'client_id': self.customer_client_id, # <-- REMOVED THIS FILTER
            "excludeclosed": "true",
            "orderby": "id",
            "orderbydesc": "true",
            "includeCustomFields": "false",
            "includedetails": "false",
            "page_size": 100,
        }
        log.info(
            f"Fetching ALL open tickets using params (NO client_id filter): {params}"
        )
        try:
            tickets_data = self._get_all_paged("/tickets", params, max_pages=max_pages)
        except PermissionError as pe:
            log.error(
                f"Permission denied fetching all tickets. API key might require client_id scoping? Error: {pe}"
            )
            # Optionally, fallback to client-scoped view if desired?
            # if self.customer_client_id:
            #    log.warning("Falling back to client-scoped ticket fetch.")
            #    params['client_id'] = self.customer_client_id
            #    tickets_data = self._get_all_paged("/tickets", params, max_pages=max_pages)
            # else:
            #    raise pe # Re-raise if no client_id to fallback to
            return []  # Return empty on permission error for now
        except Exception as e:
            log.error(f"Failed to fetch all open tickets from API: {e}", exc_info=True)
            return []

        parsed_tickets = []
        for data in tickets_data:
            try:
                # Basic data cleaning for Refs
                if isinstance(data.get("status_id"), int):
                    data["status_id"] = HaloRef(id=data["status_id"])
                elif data.get("status_id") is None:
                    data["status_id"] = HaloRef(id=DEFAULT_NEW_STATUS_ID)
                elif isinstance(data.get("status_id"), dict):
                    data["status_id"] = HaloRef(**data["status_id"])

                if isinstance(data.get("site_id"), int):
                    data["site_id"] = HaloRef(id=data["site_id"])
                elif isinstance(data.get("site_id"), dict):
                    data["site_id"] = HaloRef(**data["site_id"])
                else:
                    data["site_id"] = None

                if isinstance(data.get("categoryid_1"), int):
                    data["categoryid_1"] = HaloRef(id=data["categoryid_1"])
                elif isinstance(data.get("categoryid_1"), dict):
                    data["categoryid_1"] = HaloRef(**data["categoryid_1"])
                else:
                    data["categoryid_1"] = None

                if "summary" not in data:
                    data["summary"] = "No Summary Provided"
                if "use" not in data:
                    data["use"] = "ticket"
                if "team" not in data:
                    data["team"] = None

                ticket = Ticket(**data)
                if not ticket.status_id or ticket.status_id.id != TICKET_STATUS_CLOSED:
                    parsed_tickets.append(ticket)

            except ValidationError as val_err:
                log.error(
                    f"Validation Error parsing list ticket {data.get('id', 'N/A')}: {val_err} - Data: {data}"
                )
            except Exception as parse_error:
                log.error(
                    f"General Error parsing list ticket {data.get('id', 'N/A')}: {parse_error} - Data: {data}"
                )

        log.info(
            f"Successfully parsed {len(parsed_tickets)} open tickets from API fetch (potentially across all clients)."
        )
        return parsed_tickets

    # --- END UPDATED METHOD ---

    def get_ticket_actions(self, ticket_id: int) -> list[TicketAction]:
        if not ticket_id:
            return []
        params = {
            "ticket_id": ticket_id,
            "includeattachments": True,
            "includeagentdetails": True,
            "includehtmlnote": True,
        }
        actions_data = list(self._get_paged("/actions", params, max_pages=100))
        # ... (rest of parsing logic remains the same) ...
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
    ) -> bool:
        # Use customer_client_id if available for update operations
        if not self.customer_client_id:
            log.error("Cannot update ticket: HALO_CUSTOMER_CLIENT_ID not configured.")
            st.error("Update failed: App configuration missing customer client ID.")
            return False

        if field_name in ["status_id", "site_id", "categoryid_1"] and isinstance(
            new_value, int
        ):
            processed_value = new_value
        else:
            processed_value = new_value

        payload = {
            "id": ticket_id,
            field_name: processed_value,
            "use": "ticket",
            "client_id": self.customer_client_id,
        }
        log.info(
            f"Updating ticket {ticket_id} field '{field_name}' with payload: {payload}"
        )
        try:
            # Explicitly add client_id to the endpoint query params for POST/PUT too, sometimes required
            endpoint = f"/tickets?client_id={self.customer_client_id}"
            response = self._post(
                endpoint, [payload]
            )  # API expects list for ticket updates
            if (
                response is None
                or (
                    isinstance(response, list)
                    and response
                    and response[0].get("id") == ticket_id
                )
                or (isinstance(response, dict) and response.get("id") == ticket_id)
            ):
                log.info(f"Successfully updated '{field_name}' for ticket {ticket_id}")
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

    def add_ticket_action(self, payload: Dict[str, Any]) -> bool:
        try:
            response = self._post("/actions", [payload])  # API expects list
            # ... (rest of response check logic remains the same) ...
            if response and (
                (isinstance(response, list) and response[0].get("id") is not None)
                or (isinstance(response, dict) and response.get("id") is not None)
            ):
                log.info(
                    f"Successfully added action to ticket {payload.get('ticket_id')}"
                )
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

    def close_ticket_with_agent(self, ticket_id: int, agent_id: int = 21) -> bool:
        log.info(
            f"Attempting robust close for ticket {ticket_id} with agent {agent_id}"
        )
        # Use customer_client_id if available for close operations
        if not self.customer_client_id:
            log.error("Cannot close ticket: HALO_CUSTOMER_CLIENT_ID not configured.")
            st.error("Close failed: App configuration missing customer client ID.")
            return False
        endpoint = f"/tickets?client_id={self.customer_client_id}"
        try:
            current_ticket = self.get_ticket(
                ticket_id
            )  # get_ticket already handles client_id if set
            if not current_ticket:
                log.error(
                    f"Cannot close ticket {ticket_id}: Failed to fetch current state."
                )
                return False

            current_agent_id = current_ticket.agent_id
            needs_agent_assign = current_agent_id is None or current_agent_id <= 1
            agent_to_use_for_close = (
                agent_id if needs_agent_assign else current_agent_id
            )

            if needs_agent_assign:
                log.info(
                    f"Assigning default agent {agent_id} before closing ticket {ticket_id}."
                )
                assign_payload = [
                    {
                        "id": ticket_id,
                        "agent_id": agent_id,
                        "use": "ticket",
                        "client_id": self.customer_client_id,
                    }
                ]
                try:
                    assign_resp = self._post(endpoint, assign_payload)
                    log.info(
                        f"Agent assignment response for ticket {ticket_id}: {assign_resp}"
                    )
                    time.sleep(0.5)
                except Exception as assign_e:
                    log.warning(
                        f"Pre-closure agent assignment call failed (might still work): {assign_e}"
                    )
            else:
                log.info(
                    f"Ticket {ticket_id} already has valid agent {current_agent_id}. Proceeding to close."
                )

            close_payload = [
                {
                    "id": ticket_id,
                    "status_id": TICKET_STATUS_CLOSED,
                    "agent_id": agent_to_use_for_close,
                    "use": "ticket",
                    "client_id": self.customer_client_id,
                }
            ]
            log.info(
                f"Sending final close request for ticket {ticket_id} with payload: {close_payload}"
            )
            response = self._post(endpoint, close_payload)
            # ... (rest of response check logic remains the same) ...
            if response is None or (
                isinstance(response, list)
                and response
                and response[0].get("id") == ticket_id
            ):
                log.info(f"Successfully closed ticket {ticket_id}")
                return True
            else:
                log.error(
                    f"Final closure step failed for {ticket_id}. Response: {response}"
                )
                if (
                    isinstance(response, dict)
                    and "message" in response
                    and "Please assign this Ticket before closing it"
                    in response["message"]
                ):
                    log.critical(
                        f"CRITICAL: Closure failed for ticket {ticket_id} due to 'assign agent' error DESPITE checks/assignment attempt."
                    )
                return False
        except Exception as e:
            log.error(
                f"Exception during robust ticket closure {ticket_id}: {e}",
                exc_info=True,
            )
            return False

    def create_ticket(self, ticket_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Use customer_client_id if available for create operations
        if not self.customer_client_id:
            log.error("Cannot create ticket: HALO_CUSTOMER_CLIENT_ID not configured.")
            st.error("Create failed: App configuration missing customer client ID.")
            return {"error": "App configuration missing customer client ID."}

        ticket_data["client_id"] = self.customer_client_id
        ticket_data["use"] = "ticket"
        log.info(
            f"Attempting to create Halo ticket: {json.dumps(ticket_data, indent=2)}"
        )
        endpoint = f"/tickets?client_id={self.customer_client_id}"
        try:
            response_data = self._post(endpoint, [ticket_data])  # API expects list
            # ... (rest of response handling logic remains the same) ...
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
                error_info = response_data.get("error", "")
                log.error(f"API Error Details: {error_info}")
                return {"error": response_data["message"]}
            elif (
                isinstance(response_data, list)
                and response_data
                and response_data[0].get("id") is not None
            ):
                log.info(
                    f"Successfully created Halo ticket ID: {response_data[0]['id']}"
                )
                return response_data[0]
            elif (
                isinstance(response_data, dict) and response_data.get("id") is not None
            ):
                log.warning(
                    f"Received single object on create, expected list. ID: {response_data['id']}"
                )
                return response_data
            else:
                log.error(
                    f"Ticket creation failed or returned unexpected response: {response_data}"
                )
                return {"error": f"Unexpected response: {response_data}"}
        except Exception as e:
            log.error(f"Error creating Halo ticket: {e}", exc_info=True)
            return {"error": str(e)}

    def search_users(self, query: str, max_results: int = 50) -> list[User]:
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


# --- End Inlined HaloClient ---


# --- Streamlit App Logic ---


# Helper Functions (remain the same)
def format_datetime(dt_str: Optional[str]) -> str:
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


# Caching Functions (Update ticket fetch, others remain similar)
@st.cache_resource(ttl=300)
def search_users_cached(_halo_client: StandaloneHaloClient, query: str) -> list[User]:
    if not query or len(query.strip()) < 3:
        return []
    return _halo_client.search_users(query)


@st.cache_resource
def get_halo_client() -> StandaloneHaloClient:
    log.info("Initializing Halo Client for Streamlit session.")
    try:
        client = StandaloneHaloClient()
        client._check_token()  # Initial authentication
        return client
    except Exception as e:
        st.error(f"Fatal: Failed Halo Client initialization: {e}", icon="🚨")
        log.exception("Fatal error during Halo client initialization.")
        st.stop()


@st.cache_resource(ttl=3600)  # Cache static data longer
def get_statuses(_halo_client: StandaloneHaloClient) -> list[Status]:
    log.info("Fetching statuses...")
    try:
        return _halo_client.get_statuses()
    except Exception as e:
        st.error(f"Error fetching statuses: {e}")
        log.exception("Error fetching statuses.")
        return []


@st.cache_resource(ttl=3600)
def get_categories(_halo_client: StandaloneHaloClient) -> list[Category]:
    log.info("Fetching categories...")
    try:
        return _halo_client.get_categories()
    except Exception as e:
        st.error(f"Error fetching categories: {e}")
        log.exception("Error fetching categories.")
        return []


@st.cache_resource(ttl=3600)
def get_users(_halo_client: StandaloneHaloClient) -> list[User]:
    log.info("Fetching users (agents)...")
    try:
        return _halo_client.get_users()
    except Exception as e:
        st.error(f"Error fetching users/agents: {e}")
        log.exception("Error fetching users.")
        return []


@st.cache_resource(ttl=3600)
def get_teams(_halo_client: StandaloneHaloClient) -> list[dict]:
    log.info("Fetching teams (including inactive)...")
    try:
        return _halo_client.get_teams()
    except Exception as e:
        st.error(f"Error fetching teams: {e}")
        log.exception("Error fetching teams.")
        return []


@st.cache_resource(ttl=3600)
def get_sites(_halo_client: StandaloneHaloClient) -> list[Site]:
    log.info("Fetching sites...")
    try:
        return _halo_client.get_sites()
    except Exception as e:
        st.error(f"Error fetching sites: {e}")
        log.exception("Error fetching sites.")
        return []


@st.cache_resource(ttl=3600)
def get_ticket_types(_halo_client: StandaloneHaloClient) -> list[dict]:
    log.info("Fetching ticket types...")
    try:
        return _halo_client.get_ticket_types()
    except Exception as e:
        st.error(f"Error fetching ticket types: {e}")
        log.exception("Error fetching ticket types.")
        return []


# Updated caching function for tickets
@st.cache_resource(
    ttl=60, show_spinner="Fetching open tickets..."
)  # Cache all open tickets for 1 minute
def get_all_open_tickets_cached(
    _halo_client: StandaloneHaloClient,
    _cache_key=None,
) -> list[Ticket]:
    log.info(
        f"Fetching ALL open tickets via cache function (cache key: {_cache_key})..."
    )
    try:
        # Call the updated client method that fetches without client_id filter
        return _halo_client.get_all_open_tickets()
    except Exception as e:
        st.error(f"Error fetching all open tickets: {e}", icon="⚠️")
        log.exception("Error fetching all open tickets.")
        return []


@st.cache_resource(ttl=300, show_spinner="Fetching ticket details...")
def get_ticket_details_cached(
    _halo_client: StandaloneHaloClient, ticket_id: int, _cache_key=None
) -> Optional[Ticket]:
    if not ticket_id:
        return None
    log.info(f"Fetching ticket details for {ticket_id} (cache key: {_cache_key})...")
    try:
        return _halo_client.get_ticket(ticket_id)
    except Exception as e:
        st.error(f"Error fetching ticket details {ticket_id}: {e}")
        log.exception(f"Error fetching ticket details {ticket_id}.")
        return None


@st.cache_resource(ttl=60, show_spinner="Fetching ticket actions...")
def get_ticket_actions_cached(
    _halo_client: StandaloneHaloClient, ticket_id: int, _cache_key=None
) -> list[TicketAction]:
    if not ticket_id:
        return []
    log.info(f"Fetching ticket actions for {ticket_id} (cache key: {_cache_key})...")
    try:
        return _halo_client.get_ticket_actions(ticket_id)
    except Exception as e:
        st.error(f"Error fetching ticket actions {ticket_id}: {e}")
        log.exception(f"Error fetching ticket actions {ticket_id}.")
        return []


def clear_ticket_cache(ticket_id: Optional[int] = None) -> None:
    """Flushes relevant Streamlit caches and bumps the keys."""
    st.cache_resource.clear()
    st.cache_data.clear()
    if ticket_id:
        st.session_state[f"ticket_details_key_{ticket_id}"] = time.time()
        st.session_state[f"ticket_actions_key_{ticket_id}"] = time.time()
    st.session_state["all_open_tickets_key"] = time.time()
    log.info(f"Cleared Streamlit cache (ticket_id: {ticket_id})")


def get_status_name(halo_client: StandaloneHaloClient, status_id: Optional[int]) -> str:
    if status_id is None:
        return "N/A"
    statuses = get_statuses(halo_client)
    return next((s.name for s in statuses if s.id == status_id), f"ID:{status_id}")


def get_category_name(
    halo_client: StandaloneHaloClient, category_id: Optional[int]
) -> str:
    if category_id is None:
        return "N/A"
    categories = get_categories(halo_client)
    return next(
        (c.category_name for c in categories if c.id == category_id),
        f"ID:{category_id}",
    )


# --- UI Functions --- (Mostly similar, ensure display_ticket_list uses new cache function)
def display_ticket_list(halo_client: StandaloneHaloClient):
    st.title("Halo Ticket Interface")

    list_cols_1 = st.columns([3, 1])
    with list_cols_1[0]:
        if st.button(
            "🔄 Refresh All Tickets", key="refresh_list_top", use_container_width=True
        ):
            clear_ticket_cache()
            st.rerun()
    with list_cols_1[1]:
        if st.button(
            "➕ Create New Ticket", key="nav_create_ticket", use_container_width=True
        ):
            st.session_state.view = "create"
            st.session_state.current_ticket_id = None
            st.rerun()

    st.header("Open Tickets by Team")

    cache_key = st.session_state.get("all_open_tickets_key", time.time())
    # Use the updated cache function calling the correct client method
    all_tickets = get_all_open_tickets_cached(halo_client, _cache_key=cache_key)
    teams_data = get_teams(halo_client)  # Fetches inactive teams too now

    if not all_tickets:
        st.warning(
            "No open tickets found or permission error fetching tickets. Check logs."
        )
    else:
        # Group tickets by team
        tickets_by_team = defaultdict(list)
        active_teams_in_tickets = set()
        for t in all_tickets:
            team_name = t.team if t.team else "Unassigned"
            tickets_by_team[team_name].append(t)
            if t.team:
                active_teams_in_tickets.add(t.team)

        # Prepare data for display maps
        status_map = {s.id: s.name for s in get_statuses(halo_client)}
        category_map = {c.id: c.category_name for c in get_categories(halo_client)}

        def format_ticket_for_display(t: Ticket) -> dict:
            # ... (formatting logic remains the same) ...
            status_name = "N/A"
            if t.status_id and t.status_id.id is not None:
                status_name = status_map.get(t.status_id.id, f"ID:{t.status_id.id}")
            category_name = "N/A"
            if t.categoryid_1 and t.categoryid_1.id is not None:
                category_name = category_map.get(
                    t.categoryid_1.id, f"ID:{t.categoryid_1.id}"
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

        # Create Tabs - Filter team names to only those with tickets + Unassigned
        team_names_for_tabs = sorted(
            [name for name in tickets_by_team if name != "Unassigned"]
        )
        tab_titles = [f"All ({len(all_tickets)})"]
        if "Unassigned" in tickets_by_team:
            tab_titles.append(f"Unassigned ({len(tickets_by_team['Unassigned'])})")
        tab_titles.extend(
            [f"{name} ({len(tickets_by_team[name])})" for name in team_names_for_tabs]
        )

        tabs = st.tabs(tab_titles)
        current_tab_index = 0

        # "All" Tab
        with tabs[current_tab_index]:
            all_tickets_display = [format_ticket_for_display(t) for t in all_tickets]
            df_all = pd.DataFrame(all_tickets_display).sort_values(
                by="ID", ascending=False
            )
            st.dataframe(df_all, hide_index=True, use_container_width=True, height=600)
        current_tab_index += 1

        # "Unassigned" Tab (if exists)
        if "Unassigned" in tickets_by_team:
            with tabs[current_tab_index]:
                unassigned_tickets = tickets_by_team["Unassigned"]
                unassigned_display = [
                    format_ticket_for_display(t) for t in unassigned_tickets
                ]
                df_unassigned = pd.DataFrame(unassigned_display).sort_values(
                    by="ID", ascending=False
                )
                st.dataframe(
                    df_unassigned, hide_index=True, use_container_width=True, height=600
                )
            current_tab_index += 1

        # Team-Specific Tabs
        for team_name in team_names_for_tabs:
            with tabs[current_tab_index]:
                team_tickets = tickets_by_team[team_name]
                team_tickets_display = [
                    format_ticket_for_display(t) for t in team_tickets
                ]
                df_team = pd.DataFrame(team_tickets_display).sort_values(
                    by="ID", ascending=False
                )
                st.dataframe(
                    df_team, hide_index=True, use_container_width=True, height=600
                )
            current_tab_index += 1

        # Display inactive teams found by API but having no *open* tickets
        inactive_teams_found = {t["name"] for t in teams_data if t.get("inactive")}
        teams_without_open_tickets = (
            inactive_teams_found - active_teams_in_tickets - {"Unassigned"}
        )
        if teams_without_open_tickets:
            st.caption(
                f"Note: The following teams exist but have no open tickets assigned: {', '.join(sorted(list(teams_without_open_tickets)))}"
            )

    # --- Section to Open Specific Ticket by ID --- (remains the same)
    st.write("---")
    st.subheader("Open Ticket by ID")
    open_ticket_cols = st.columns([1, 3])
    with open_ticket_cols[0]:
        selected_id_input = st.number_input(
            "Ticket ID:",
            min_value=1,
            step=1,
            value=None,
            key="open_ticket_input_main",
            label_visibility="collapsed",
        )
    with open_ticket_cols[1]:
        if (
            st.button("Open Ticket", key="open_ticket_button_main")
            and selected_id_input
        ):
            st.session_state.view = "detail"
            st.session_state.current_ticket_id = selected_id_input
            clear_ticket_cache(selected_id_input)
            st.rerun()


# --- display_ticket_detail Function --- (largely unchanged, ensure it uses cached data correctly)
def display_ticket_detail(halo_client: StandaloneHaloClient, ticket_id: int):
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

    col1, col2 = st.columns([3, 1])  # Main content | Actions sidebar

    # Sidebar Actions
    with col2:
        st.subheader("Actions")
        if st.button(
            "⬅️ Back to List", key="back_button_detail", use_container_width=True
        ):
            st.session_state.view = "list"
            st.session_state.current_ticket_id = None
            st.rerun()
        if st.button(
            "🔄 Refresh Details", key="refresh_detail_button", use_container_width=True
        ):
            clear_ticket_cache(ticket_id)
            st.rerun()
        st.write("---")
        current_status_id = ticket.status_id.id if ticket and ticket.status_id else None
        if ticket and current_status_id != TICKET_STATUS_CLOSED:
            if st.button(
                "🔒 Close Ticket",
                key="close_ticket_button_detail",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Closing ticket..."):
                    # Ensure agent_id is appropriate (e.g., logged in user's ID if available, or default)
                    ok = halo_client.close_ticket_with_agent(
                        ticket_id, agent_id=21
                    )  # Default agent 21
                if ok:
                    st.success("Ticket closed.")
                    clear_ticket_cache(ticket_id)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to close ticket.")
        elif ticket:
            st.info("Ticket is already closed.")

    # Main Content
    with col1:
        if not ticket:
            st.error(f"Could not load details for ticket {ticket_id}.", icon="❌")
            return  # Back button is in sidebar

        st.subheader("Ticket Details")
        info_cols = st.columns(3)
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

        st.subheader("Classification & Assignment")
        field_cols = st.columns(4)
        # Status Dropdown
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
        # Category Dropdown
        with field_cols[1]:
            categories = get_categories(halo_client)
            category_options = {c.id: c.category_name for c in categories}
            current_category_id = (
                ticket.categoryid_1.id if ticket.categoryid_1 else 0
            )  # Treat None as 0 for comparison
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
                        # API likely expects null or the ID, 0 might not work for 'None'. Let's send null if 0 selected.
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
        # Agent Dropdown
        with field_cols[2]:
            agents = get_users(halo_client)
            agent_options = {a.id: a.name for a in agents if a.name}
            current_agent_id = ticket.agent_id or 0  # Treat None as 0
            if (
                current_agent_id != 0 and current_agent_id not in agent_options
            ):  # Add current if missing
                agent_name = next(
                    (
                        u.name
                        for u in get_users(halo_client)
                        if u.id == current_agent_id
                    ),
                    f"ID:{current_agent_id}",
                )
                if agent_name:
                    agent_options[current_agent_id] = agent_name
            agent_options_with_none = {0: "--- Unassigned ---", **agent_options}
            selected_agent_id = st.selectbox(
                "Agent",
                options=list(agent_options_with_none.keys()),
                format_func=lambda x: agent_options_with_none[x],
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
                        # API expects the ID, or 0 for unassigned.
                        if halo_client.update_ticket_field(
                            ticket_id, "agent_id", selected_agent_id
                        ):
                            clear_ticket_cache(ticket_id)
                            st.rerun()
                        else:
                            st.error("Update failed.")
        # Team Dropdown
        with field_cols[3]:
            teams = get_teams(halo_client)
            team_options = {
                t["name"]: t["name"]
                for t in teams
                if t.get("name") and not t.get("inactive")
            }  # Show only active teams for assignment
            current_team_name = ticket.team or ""
            if current_team_name and current_team_name not in team_options:
                team_options[current_team_name] = (
                    f"Current ({current_team_name})"  # Add current even if inactive
                )
            team_options_with_none = {"": "--- Unassigned ---", **team_options}
            selected_team_name = st.selectbox(
                "Team",
                options=list(team_options_with_none.keys()),
                format_func=lambda x: team_options_with_none[x],
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
                        # API expects the name string, or "" / null for unassigned.
                        if halo_client.update_ticket_field(
                            ticket_id, "team", selected_team_name or None
                        ):
                            clear_ticket_cache(ticket_id)
                            st.rerun()
                        else:
                            st.error("Update failed.")

        # Details, History/Notes, Add Note sections remain the same conceptually
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
                    # Email headers rendering...
                    header_lines = []
                    if action.emailfrom:
                        header_lines.append(f"**From:** {action.emailfrom}")
                    # ... (rest of headers) ...
                    if header_lines:
                        st.markdown("<br>".join(header_lines), unsafe_allow_html=True)
                        st.markdown("---")
                    # Body rendering...
                    if action.note_html:
                        safe_note_html = action.note_html.replace("<script", "<script")
                        st.markdown(safe_note_html, unsafe_allow_html=True)
                    elif action.note:
                        st.text(action.note)
                    else:
                        st.markdown("_No note content._")

        st.subheader("Add New Note / Email User")
        note_key = f"new_note_text_{ticket_id}"
        note_text = st.session_state.get(note_key, "")
        note_text = st.text_area(
            "Note / Email Content:",
            value=note_text,
            key=f"text_area_{note_key}",
            height=150,
        )
        st.session_state[note_key] = note_text
        note_cols = st.columns(2)
        with note_cols[0]:
            if st.button(
                "Add Private Note",
                key=f"add_private_note_{ticket_id}",
                use_container_width=True,
            ):
                if note_text.strip():
                    with st.spinner("Adding note..."):
                        if add_new_note(
                            halo_client, ticket_id, note_text, is_private=True
                        ):
                            st.session_state[note_key] = ""
                            st.rerun()
                else:
                    st.warning("Enter note content.")
        with note_cols[1]:
            if st.button(
                "✉️ Add Note & Email User",
                key=f"email_user_{ticket_id}",
                use_container_width=True,
            ):
                if note_text.strip():
                    with st.spinner("Sending..."):
                        if add_new_note(
                            halo_client,
                            ticket_id,
                            note_text,
                            is_private=False,
                            send_email=True,
                        ):
                            st.session_state[note_key] = ""
                            st.rerun()
                else:
                    st.warning("Enter email content.")


# --- add_new_note Function --- (remains the same)
def add_new_note(
    halo_client: StandaloneHaloClient,
    ticket_id: int,
    note_content: str,
    is_private: bool = True,
    send_email: bool = False,
) -> bool:
    outcome = (
        "Private Note" if is_private else ("Email Sent" if send_email else "Note Added")
    )
    log.info(f"Adding action '{outcome}' to ticket {ticket_id}")
    note_html = f"<p>{note_content.replace(chr(10), '<br>')}</p>"
    payload = {
        "ticket_id": ticket_id,
        "outcome": outcome,
        "note": note_content,
        "note_html": note_html,
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
            log.info(f"Email will be sent to: {ticket.user_email}")
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


# --- display_create_ticket_form Function --- (Needs to ensure client_id is optional or handled)
def display_create_ticket_form(halo_client: StandaloneHaloClient):
    st.title("Create New Halo Ticket")
    if st.button("⬅️ Back to List", key="back_button_create"):
        st.session_state.view = "list"
        st.session_state.current_ticket_id = None
        st.rerun()
    st.write("---")

    # Check if customer_client_id is configured, as it's needed for creation
    if not halo_client.customer_client_id:
        st.error(
            "Ticket creation unavailable: App configuration missing `HALO_CUSTOMER_CLIENT_ID`.",
            icon="🚫",
        )
        return

    # Load form data (cached)
    with st.spinner("Loading form data..."):
        statuses = get_statuses(halo_client)
        categories = get_categories(halo_client)
        sites = get_sites(halo_client)
        teams = [
            t for t in get_teams(halo_client) if not t.get("inactive")
        ]  # Active teams for assignment
        types = get_ticket_types(halo_client)
        agents = [u for u in get_users(halo_client) if not u.inactive and u.name]

    # Create form options maps...
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
        # User selection logic... (remains similar)
        selected_user_display = "No user selected."
        if st.session_state.create_form_selected_user:
            selected_user_display = f"Selected: {st.session_state.create_form_selected_user.name} ({st.session_state.create_form_selected_user.emailaddress})"
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
            missing.append("Reporter (Search and select)")
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
            # Add custom fields payload if needed here

            with st.spinner("Creating ticket..."):
                result = halo_client.create_ticket(payload)
            if result and result.get("id"):
                new_id = result["id"]
                st.success(f"Ticket #{new_id} created!")
                st.session_state.create_form_selected_user = (
                    None  # Clear user selection
                )
                clear_ticket_cache()  # Clear list cache
                st.session_state.view = "detail"
                st.session_state.current_ticket_id = new_id
                st.rerun()
            else:
                error_msg = result.get("error", "Unknown error.")
                st.error(f"Creation failed: {error_msg}")
                log.error(
                    f"Ticket creation failed. Payload: {payload}, Response: {result}"
                )


# --- Main App ---
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
    else:  # Default back to list view
        st.session_state.view = "list"
        st.session_state.current_ticket_id = None
        st.rerun()
