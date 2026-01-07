#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive script to authenticate with CDSE openEO.
This will guide you through the OIDC device code authentication process.
"""

import logging
import sys
import time

import openeo

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise from openeo library
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Create a custom logger for our messages
print_logger = logging.getLogger("user")
print_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
print_logger.addHandler(handler)


def authenticate_openeo():
    """Authenticate with CDSE openEO using device code flow."""
    endpoint = "openeo.dataspace.copernicus.eu"
    
    print("\n" + "=" * 70)
    print("CDSE openEO Authentication")
    print("=" * 70)
    print(f"Endpoint: {endpoint}")
    print("")
    print("This script uses device code authentication.")
    print("You will need to:")
    print("  1. Visit a URL in your browser (shown below)")
    print("  2. Log in with your CDSE account credentials")
    print("  3. Enter the device code shown below")
    print("  4. Authorize the application")
    print("")
    print("If you don't have a CDSE account, register at:")
    print("  https://identity.dataspace.copernicus.eu/")
    print("")
    print("=" * 70)
    print("Starting authentication...")
    print("=" * 70 + "\n")
    
    try:
        # Connect to openEO
        connection = openeo.connect(endpoint)
        
        # Start authentication - this will trigger device code flow
        print("Initiating device code authentication...")
        print("Please wait for the authentication URL and device code...")
        print("")
        
        # Authenticate - this will print the URL and code
        connection.authenticate_oidc()
        
        # If we get here, authentication was successful
        print("\n" + "=" * 70)
        print("✓ Authentication successful!")
        print("=" * 70)
        
        # Test the connection
        try:
            capabilities = connection.capabilities()
            print(f"API version: {capabilities.api_version()}")
            print(f"Backend: {capabilities.title()}")
        except Exception as e:
            print(f"Note: Could not retrieve capabilities ({e})")
        
        print("")
        print("Your credentials have been saved.")
        print("You can now run EFAST with --data-source openeo")
        print("=" * 70 + "\n")
        
        return True
        
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Authentication cancelled by user")
        print("=" * 70 + "\n")
        return False
    except Exception as e:
        error_msg = str(e)
        print("\n" + "=" * 70)
        print("✗ Authentication failed")
        print("=" * 70)
        print(f"Error: {error_msg}")
        print("")
        
        if "Timeout" in error_msg or "authorization_pending" in error_msg:
            print("The authentication timed out.")
            print("This usually means:")
            print("  - You didn't visit the URL in time")
            print("  - You didn't complete the authorization")
            print("  - The device code expired (they expire after ~5 minutes)")
            print("")
            print("Please run this script again and complete the authentication")
            print("within 5 minutes of seeing the URL and device code.")
        else:
            print("Troubleshooting:")
            print("  1. Ensure you have a CDSE account")
            print("     Register at: https://identity.dataspace.copernicus.eu/")
            print("  2. Check your internet connection")
            print("  3. Make sure you visit the URL and complete authorization")
        
        print("=" * 70 + "\n")
        return False


if __name__ == "__main__":
    success = authenticate_openeo()
    sys.exit(0 if success else 1)






