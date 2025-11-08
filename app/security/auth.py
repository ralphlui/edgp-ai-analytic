from fastapi import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import httpx
import logging
from app.config import ADMIN_API_BASE_URL, JWT_SECRET_KEY, JWT_ALGORITHM
from app.security.pii_redactor import PIIRedactionFilter, redact_pii

bearer_scheme = HTTPBearer()

logger = logging.getLogger(__name__)

# Add PII redaction filter to this logger
pii_filter = PIIRedactionFilter()
logger.addFilter(pii_filter)

# Validate required configuration
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY is required but not found in environment variables")

if not ADMIN_API_BASE_URL:
    raise ValueError("ADMIN_API_BASE_URL is required but not found in environment variables")

def validate_jwt_token(credentials: HTTPAuthorizationCredentials):
    """
    Validate JWT token and extract payload.
    
    Args:
        credentials: HTTP authorization credentials containing the JWT token
        
    Returns:
        dict: JWT payload containing user information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.error(f"JWT validation failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def validate_user_profile_with_response(credentials: HTTPAuthorizationCredentials):
    """
    Validate JWT token and check user profile status via admin API.
    Returns a structured response instead of raising exceptions for inactive users.
    
    Args:
        credentials: HTTP authorization credentials containing the JWT token
        
    Returns:
        dict: Response with success, message, chart_image, and payload (if successful)
    """
    try:
        # First validate the JWT token
        payload = validate_jwt_token(credentials)
    except HTTPException as e:
        return {
            "success": False,
            "message": f"Authentication failed: {e.detail}",
            "chart_image": None
        }
    
    # Extract user ID from JWT payload (sub claim)
    user_id = payload.get("sub")
    if not user_id:
        logger.error("No user ID found in JWT token (sub claim missing)")
        return {
            "success": False,
            "message": "Invalid token: missing user ID",
            "chart_image": None
        }
    
    # Call admin API to verify user profile
    try:
        profile_url = f"{ADMIN_API_BASE_URL}/users/profile"
        headers = {
            "Authorization": f"Bearer {credentials.credentials}",
            "X-User-Id": str(user_id),
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Validating user profile for user_id:")
            response = await client.get(profile_url, headers=headers)
            
            if response.status_code == 200:
                profile_data = response.json()
                
                # Check if user is active
                logger.info(f"Profile data for user {user_id}: {profile_data}")
                if profile_data.get("success") is True:
                    logger.info(f"User {user_id} is active and validated")
                    return {
                        "success": True,
                        "message": "User authenticated and active",
                        "chart_image": None,
                        "payload": payload
                    }
                else:
                    #logger.warning(f"User {user_id} is not active: {profile_data.get('success')}")
                    return {
                        "success": False,
                        "message": "User account is not active. Please contact your administrator to activate your account.",
                        "chart_image": None
                    }
            
            elif response.status_code == 401:
                # For 401, check if there's a JSON response with error details
                try:
                    logger.info(f"401 response text for user {user_id}: {response.text}")
                    error_data = response.json()
                    error_message = error_data.get("message", "Authentication failed")
                    logger.warning(f"Admin API returned 401 for user {user_id}: {error_message}")
                    return {
                        "success": False,
                        "message": error_message,
                        "chart_image": None
                    }
                except Exception as parse_error:
                    # Fallback if response is not JSON
                    logger.error(f"Failed to parse 401 response for user {user_id}: {parse_error}")
                    logger.error(f"Response text: {response.text}")
                    return {
                        "success": False,
                        "message": "Authentication failed: Invalid credentials",
                        "chart_image": None
                    }
            
            elif response.status_code == 404:
                logger.error(f"User {user_id} not found in admin system")
                return {
                    "success": False,
                    "message": "User not found in the system",
                    "chart_image": None
                }
            
            else:
                logger.error(f"Admin API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "message": "Unable to verify user profile. Please try again later.",
                    "chart_image": None
                }
                
    except httpx.TimeoutException:
        logger.error(f"Timeout calling admin API for user")
        return {
            "success": False,
            "message": "User profile validation timeout. Please try again.",
            "chart_image": None
        }
    except httpx.RequestError as e:
        logger.error(f"Request error calling admin API: {e}")
        return {
            "success": False,
            "message": "Unable to connect to user profile service. Please check your connection.",
            "chart_image": None
        }
    except Exception as e:
        logger.error(f"Unexpected error during user profile validation: {e}")
        return {
            "success": False,
            "message": "User profile validation failed due to an unexpected error.",
            "chart_image": None
        }


async def validate_user_profile(credentials: HTTPAuthorizationCredentials):
    """
    Validate JWT token and check user profile status via admin API.
    
    Args:
        credentials: HTTP authorization credentials containing the JWT token
        
    Returns:
        dict: JWT payload if user is valid and active
        
    Raises:
        HTTPException: If token is invalid, user not found, or user is inactive
    """
    # First validate the JWT token
    payload = validate_jwt_token(credentials)
    
    # Extract user ID from JWT payload (sub claim)
    user_id = payload.get("sub")
    if not user_id:
        logger.error("No user ID found in JWT token (sub claim missing)")
        raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
    
    # Call admin API to verify user profile
    try:
        profile_url = f"{ADMIN_API_BASE_URL}/users/profile"
        headers = {
            "Authorization": f"Bearer {credentials.credentials}",
            "X-User-Id": str(user_id),
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Validating user profile for user_id")
            response = await client.get(profile_url, headers=headers)
            
            if response.status_code == 200:
                profile_data = response.json()
                
                # Check if user is active
                if profile_data.get("success") is True:
                    logger.info(f"User {user_id} is active and validated")
                    return payload
                else:
                    logger.warning(f"User {user_id} is not active: {profile_data.get('active')}")
                    raise HTTPException(
                        status_code=403, 
                        detail="User account is not active"
                    )
            
            elif response.status_code == 401:
                # For 401, check if there's a JSON response with error details
                try:
                    logger.info(f"401 response text for user")
                    error_data = response.json()
                    error_message = error_data.get("message", "Authentication failed")
                    logger.warning(f"Admin API returned 401 for user")
                    raise HTTPException(status_code=403, detail=error_message)
                except Exception as parse_error:
                    # Fallback if response is not JSON
                    logger.error(f"Failed to parse 401 response for user {parse_error}")
                    logger.error(f"Response text: {response.text}")
                    raise HTTPException(status_code=401, detail="Authentication failed")
            
            elif response.status_code == 404:
                logger.error(f"User not found in admin system")
                raise HTTPException(status_code=404, detail="User not found")
            
            else:
                logger.error(f"Admin API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=500, 
                    detail="Unable to verify user profile"
                )
                
    except httpx.TimeoutException:
        logger.error(f"Timeout calling admin API for user")
        raise HTTPException(
            status_code=500, 
            detail="User profile validation timeout"
        )
    except httpx.RequestError as e:
        logger.error(f"Request error calling admin API: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Unable to connect to user profile service"
        )
    except Exception as e:
        logger.error(f"Unexpected error during user profile validation: {e}")
        raise HTTPException(
            status_code=500, 
            detail="User profile validation failed"
        )
