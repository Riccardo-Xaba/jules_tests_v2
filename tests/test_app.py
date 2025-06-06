import pytest
import numpy as np
import json
from app import app  # Assuming your Flask app instance is named 'app' in app.py

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """Test that the index page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"<title>Stitch Design</title>" in response.data # Or other relevant assertion

def test_calculate_transformation_matrix(client):
    # Test case 1: Zero rotation, zero translation (Identity matrix)
    data = {
        "x": 0, "y": 0, "z": 0,
        "rx": 0, "ry": 0, "rz": 0
    }
    response = client.post('/calculate', data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200

    # Ensure response is valid JSON (get_json() would raise an error if not)
    # and then convert to numpy array
    result_matrix = np.array(response.get_json())
    expected_matrix = np.identity(4)

    # Using numpy.testing.assert_allclose for float comparisons
    np.testing.assert_allclose(result_matrix, expected_matrix, atol=1e-7)

    # Test case 2: Z-axis rotation only
    data_z_rot = {
        "x": 0, "y": 0, "z": 0,
        "rx": 0, "ry": 0, "rz": 90
    }
    response_z_rot = client.post('/calculate', data=json.dumps(data_z_rot), content_type='application/json')
    assert response_z_rot.status_code == 200
    result_matrix_z_rot = np.array(response_z_rot.get_json())
    expected_matrix_z_rot = np.array([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    np.testing.assert_allclose(result_matrix_z_rot, expected_matrix_z_rot, atol=1e-7)

    # Test case 3: X-axis rotation only
    data_x_rot = {
        "x": 0, "y": 0, "z": 0,
        "rx": 90, "ry": 0, "rz": 0
    }
    response_x_rot = client.post('/calculate', data=json.dumps(data_x_rot), content_type='application/json')
    assert response_x_rot.status_code == 200
    result_matrix_x_rot = np.array(response_x_rot.get_json())
    expected_matrix_x_rot = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ])
    np.testing.assert_allclose(result_matrix_x_rot, expected_matrix_x_rot, atol=1e-7)

    # Test case 4: Y-axis rotation only
    data_y_rot = {
        "x": 0, "y": 0, "z": 0,
        "rx": 0, "ry": 90, "rz": 0
    }
    response_y_rot = client.post('/calculate', data=json.dumps(data_y_rot), content_type='application/json')
    assert response_y_rot.status_code == 200
    result_matrix_y_rot = np.array(response_y_rot.get_json())
    expected_matrix_y_rot = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1,0, 0, 0],
        [0, 0, 0, 1]
    ])
    np.testing.assert_allclose(result_matrix_y_rot, expected_matrix_y_rot, atol=1e-7)

    # Test case 5: Translation only
    data_trans = {
        "x": 10, "y": 20, "z": 30,
        "rx": 0, "ry": 0, "rz": 0
    }
    response_trans = client.post('/calculate', data=json.dumps(data_trans), content_type='application/json')
    assert response_trans.status_code == 200
    result_matrix_trans = np.array(response_trans.get_json())
    expected_matrix_trans = np.array([
        [1, 0, 0, 10],
        [0, 1, 0, 20],
        [0, 0, 1, 30],
        [0, 0, 0,  1]
    ])
    np.testing.assert_allclose(result_matrix_trans, expected_matrix_trans, atol=1e-7)

    # Test case 6: Combined translation and Z-axis rotation
    data_combo = {
        "x": 10, "y": 20, "z": 30,
        "rx": 0, "ry": 0, "rz": 90
    }
    response_combo = client.post('/calculate', data=json.dumps(data_combo), content_type='application/json')
    assert response_combo.status_code == 200
    result_matrix_combo = np.array(response_combo.get_json())
    expected_matrix_combo = np.array([
        [0, -1, 0, 10],
        [1,  0, 0, 20],
        [0,  0, 1, 30],
        [0,  0, 0,  1]
    ])
    np.testing.assert_allclose(result_matrix_combo, expected_matrix_combo, atol=1e-7)

def test_calculate_invalid_input(client):
    # Test case: Invalid (non-numeric) input for x
    data = {
        "x": "not-a-number", "y": "0", "z": "0",
        "rx": "0", "ry": "0", "rz": "0"
    }
    response = client.post('/calculate', data=json.dumps(data), content_type='application/json')
    assert response.status_code == 400

    response_data = response.get_json()
    assert "error" in response_data
    # Assert the content of the error message for more specific testing
    assert "Invalid input data" in response_data["error"]

# Placeholder for future tests
