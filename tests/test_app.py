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

def test_calculate_transformation_matrix_diverse_inputs(client):
    # Test Case 1: Various non-zero position values
    data_trans_float = {
        "x": 1.5, "y": -2.25, "z": 3.75,
        "rx": 0, "ry": 0, "rz": 0
    }
    response_trans_float = client.post('/calculate', data=json.dumps(data_trans_float), content_type='application/json')
    assert response_trans_float.status_code == 200
    result_matrix_trans_float = np.array(response_trans_float.get_json())
    expected_matrix_trans_float = np.array([
        [1, 0, 0,  1.5],
        [0, 1, 0, -2.25],
        [0, 0, 1,  3.75],
        [0, 0, 0,  1]
    ])
    np.testing.assert_allclose(result_matrix_trans_float, expected_matrix_trans_float, atol=1e-7)

    # Test Case 2: Rotation around Z by 30 degrees
    rz_30_deg = 30
    rz_30_rad = np.radians(rz_30_deg)
    cos_30 = np.cos(rz_30_rad)
    sin_30 = np.sin(rz_30_rad)
    data_z_rot_30 = {
        "x": 0, "y": 0, "z": 0,
        "rx": 0, "ry": 0, "rz": rz_30_deg
    }
    response_z_rot_30 = client.post('/calculate', data=json.dumps(data_z_rot_30), content_type='application/json')
    assert response_z_rot_30.status_code == 200
    result_matrix_z_rot_30 = np.array(response_z_rot_30.get_json())
    expected_matrix_z_rot_30 = np.array([
        [cos_30, -sin_30, 0, 0],
        [sin_30,  cos_30, 0, 0],
        [0,       0,      1, 0],
        [0,       0,      0, 1]
    ])
    np.testing.assert_allclose(result_matrix_z_rot_30, expected_matrix_z_rot_30, atol=1e-7)

    # Test Case 3: Rotation around Y by 45 degrees
    ry_45_deg = 45
    ry_45_rad = np.radians(ry_45_deg)
    cos_45 = np.cos(ry_45_rad)
    sin_45 = np.sin(ry_45_rad)
    data_y_rot_45 = {
        "x": 0, "y": 0, "z": 0,
        "rx": 0, "ry": ry_45_deg, "rz": 0
    }
    response_y_rot_45 = client.post('/calculate', data=json.dumps(data_y_rot_45), content_type='application/json')
    assert response_y_rot_45.status_code == 200
    result_matrix_y_rot_45 = np.array(response_y_rot_45.get_json())
    expected_matrix_y_rot_45 = np.array([
        [cos_45,  0, sin_45, 0],
        [0,       1, 0,      0],
        [-sin_45, 0, cos_45, 0],
        [0,       0, 0,      1]
    ])
    np.testing.assert_allclose(result_matrix_y_rot_45, expected_matrix_y_rot_45, atol=1e-7)

    # Test Case 4: Combined diverse translation and rotation (Z-axis by 60 deg)
    rz_60_deg = 60
    rz_60_rad = np.radians(rz_60_deg)
    cos_60 = np.cos(rz_60_rad)
    sin_60 = np.sin(rz_60_rad)
    data_combo_60 = {
        "x": 5, "y": -10, "z": 15,
        "rx": 0, "ry": 0, "rz": rz_60_deg
    }
    response_combo_60 = client.post('/calculate', data=json.dumps(data_combo_60), content_type='application/json')
    assert response_combo_60.status_code == 200
    result_matrix_combo_60 = np.array(response_combo_60.get_json())
    expected_matrix_combo_60 = np.array([
        [cos_60, -sin_60, 0,  5],
        [sin_60,  cos_60, 0, -10],
        [0,       0,      1,  15],
        [0,       0,      0,  1]
    ])
    np.testing.assert_allclose(result_matrix_combo_60, expected_matrix_combo_60, atol=1e-7)

    # Test Case 5: Specific complex combined translation and rotation
    data_specific_combo = {
        "x": 10, "y": 200, "z": -10,
        "rx": -13.2, "ry": 77.6, "rz": 90
    }
    response_specific_combo = client.post('/calculate', data=json.dumps(data_specific_combo), content_type='application/json')
    assert response_specific_combo.status_code == 200
    result_matrix_specific_combo = np.array(response_specific_combo.get_json())

    # Calculate expected matrix
    rx_rad = np.radians(-13.2)
    ry_rad = np.radians(77.6)
    rz_rad = np.radians(90)

    R_x = np.array([[1, 0,           0],
                    [0, np.cos(rx_rad), -np.sin(rx_rad)],
                    [0, np.sin(rx_rad),  np.cos(rx_rad)]])

    R_y = np.array([[np.cos(ry_rad),  0, np.sin(ry_rad)],
                    [0,            1, 0],
                    [-np.sin(ry_rad), 0, np.cos(ry_rad)]])

    R_z = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0],
                    [np.sin(rz_rad),  np.cos(rz_rad), 0],
                    [0,            0,           1]])

    R = R_z @ R_y @ R_x

    expected_matrix_specific_combo = np.identity(4)
    expected_matrix_specific_combo[:3, :3] = R
    expected_matrix_specific_combo[:3, 3] = [10, 200, -10]

    np.testing.assert_allclose(result_matrix_specific_combo, expected_matrix_specific_combo, atol=1e-7)

# Placeholder for future tests
