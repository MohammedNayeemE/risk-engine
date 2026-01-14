"""
Example clients for the Risk Engine API.
Demonstrates how to consume the endpoints.
"""

import base64
import requests
from typing import Optional
import asyncio
from httpx import AsyncClient


class RiskEngineClient:
    """Synchronous client for Risk Engine API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_status(self) -> dict:
        """Get API status"""
        response = self.session.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def assess_with_image_file(
        self,
        image_path: str,
        name: str,
        age: str,
        gender: str,
        allergies: Optional[list[str]] = None,
        known_conditions: Optional[list[str]] = None,
    ) -> dict:
        """
        Assess risk using an image file.
        
        Args:
            image_path: Path to prescription image
            name: Patient name
            age: Patient age
            gender: Patient gender (M/F)
            allergies: List of allergies
            known_conditions: List of known conditions
            
        Returns:
            Assessment results
        """
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {
                "name": name,
                "age": age,
                "gender": gender,
                "allergies": ",".join(allergies or []),
                "known_conditions": ",".join(known_conditions or []),
            }
            response = self.session.post(
                f"{self.base_url}/assess/file",
                files=files,
                data=data,
            )
        
        response.raise_for_status()
        return response.json()
    
    def assess_with_base64(
        self,
        image_base64: str,
        name: str,
        age: str,
        gender: str,
        allergies: Optional[list[str]] = None,
        known_conditions: Optional[list[str]] = None,
        image_mime: str = "image/jpeg",
    ) -> dict:
        """
        Assess risk using base64 encoded image.
        
        Args:
            image_base64: Base64 encoded image
            name: Patient name
            age: Patient age
            gender: Patient gender (M/F)
            allergies: List of allergies
            known_conditions: List of known conditions
            image_mime: MIME type of the image
            
        Returns:
            Assessment results
        """
        payload = {
            "user_profile": {
                "name": name,
                "age": age,
                "gender": gender,
                "allergies": allergies or [],
                "known_conditions": known_conditions or [],
            },
            "image_base64": image_base64,
            "image_mime": image_mime,
        }
        response = self.session.post(
            f"{self.base_url}/assess/json",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    def batch_assess(self, requests_list: list[dict]) -> dict:
        """
        Perform batch assessment.
        
        Args:
            requests_list: List of assessment requests
            
        Returns:
            Batch results
        """
        response = self.session.post(
            f"{self.base_url}/assess/batch",
            json=requests_list,
        )
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class AsyncRiskEngineClient:
    """Asynchronous client for Risk Engine API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = None
    
    async def __aenter__(self):
        self.client = AsyncClient(base_url=self.base_url)
        return self
    
    async def __aexit__(self, *args):
        await self.client.aclose()
    
    async def health_check(self) -> dict:
        """Check API health"""
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def assess_with_base64(
        self,
        image_base64: str,
        name: str,
        age: str,
        gender: str,
        allergies: Optional[list[str]] = None,
        known_conditions: Optional[list[str]] = None,
        image_mime: str = "image/jpeg",
    ) -> dict:
        """Assess risk using base64 encoded image (async)"""
        payload = {
            "user_profile": {
                "name": name,
                "age": age,
                "gender": gender,
                "allergies": allergies or [],
                "known_conditions": known_conditions or [],
            },
            "image_base64": image_base64,
            "image_mime": image_mime,
        }
        response = await self.client.post("/assess/json", json=payload)
        response.raise_for_status()
        return response.json()


# ============================================================================
# Usage Examples
# ============================================================================

def example_basic_assessment():
    """Example: Basic assessment with file upload"""
    print("Example 1: Basic Assessment with File Upload")
    print("-" * 50)
    
    with RiskEngineClient() as client:
        # Check health first
        health = client.health_check()
        print(f"API Health: {health['status']}")
        
        # Perform assessment
        result = client.assess_with_image_file(
            image_path="/path/to/prescription.jpg",
            name="John Doe",
            age="45",
            gender="M",
            allergies=["Penicillin"],
            known_conditions=["Diabetes"],
        )
        
        print(f"\nAssessment Results:")
        print(f"User: {result['user_name']}")
        print(f"Medicines Found: {result['medicines_found']}")
        print(f"Valid: {result['is_valid']}")
        print(f"Medicines: {result['medicines']}")


def example_base64_assessment():
    """Example: Assessment with base64 encoded image"""
    print("\nExample 2: Assessment with Base64 Image")
    print("-" * 50)
    
    # Read and encode image
    with open("/path/to/prescription.jpg", "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    with RiskEngineClient() as client:
        result = client.assess_with_base64(
            image_base64=image_base64,
            name="Jane Smith",
            age="35",
            gender="F",
            allergies=["Aspirin", "Ibuprofen"],
            known_conditions=["Hypertension"],
            image_mime="image/jpeg",
        )
        
        print(f"Assessment Status: {result['status']}")
        print(f"Image Hash: {result['image_hash']}")
        print(f"Image Quality: {result['image_quality']}")


def example_batch_processing():
    """Example: Batch processing multiple assessments"""
    print("\nExample 3: Batch Processing")
    print("-" * 50)
    
    # Prepare multiple requests
    requests_list = []
    for i in range(3):
        with open(f"/path/to/prescription_{i}.jpg", "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        requests_list.append({
            "user_profile": {
                "name": f"Patient {i+1}",
                "age": str(30 + i*5),
                "gender": "M" if i % 2 == 0 else "F",
                "allergies": [],
                "known_conditions": [],
            },
            "image_base64": image_base64,
            "image_mime": "image/jpeg",
        })
    
    with RiskEngineClient() as client:
        results = client.batch_assess(requests_list)
        
        print(f"Batch Size: {results['batch_size']}")
        for item in results['results']:
            print(f"  Item {item['index']}: {item['status']}")


async def example_async_assessment():
    """Example: Async assessment"""
    print("\nExample 4: Async Assessment")
    print("-" * 50)
    
    with open("/path/to/prescription.jpg", "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    async with AsyncRiskEngineClient() as client:
        health = await client.health_check()
        print(f"API Health: {health['status']}")
        
        result = await client.assess_with_base64(
            image_base64=image_base64,
            name="Async Patient",
            age="50",
            gender="M",
        )
        
        print(f"Assessment Status: {result['status']}")
        print(f"Medicines Found: {result['medicines_found']}")


def example_error_handling():
    """Example: Error handling"""
    print("\nExample 5: Error Handling")
    print("-" * 50)
    
    client = RiskEngineClient("http://localhost:8000")
    
    try:
        # Invalid base64
        result = client.assess_with_base64(
            image_base64="invalid-base64-data",
            name="Test",
            age="30",
            gender="M",
        )
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Details: {e.response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    # Note: Ensure the API is running before running these examples
    
    try:
        # Uncomment examples to run them
        # example_basic_assessment()
        # example_base64_assessment()
        # example_batch_processing()
        # asyncio.run(example_async_assessment())
        # example_error_handling()
        
        print("Examples ready! Uncomment the examples you want to run.")
        print("Make sure the API is running: uvicorn app.main:app --reload")
        
    except Exception as e:
        print(f"Error: {e}")
