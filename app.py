# main.py
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

@app.after_request
def after_request(response):
    """
    Adds CORS headers to every response to allow the frontend to communicate with this backend.
    """
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def index():
    """
    Serves the main HTML file of the application.
    Assumes 'frontend.html' is in the same directory as this script.
    """
    return send_from_directory('.', 'frontend.html')

# --- API Endpoint for Matrix Calculation ---
# The following block defines the /calculate endpoint.
# It listens for POST requests from the frontend,
# processes the input data, and returns the calculated matrix.
@app.route('/calculate', methods=['POST', 'OPTIONS'])
def calculate_transformation_matrix():
    """
    Calculates the homogeneous transformation matrix based on position and Euler angles.
    The request body should be a JSON object containing:
    {
        "x": float,
        "y": float,
        "z": float,
        "rx": float, # Euler angle in degrees
        "ry": float, # Euler angle in degrees
        "rz": float  # Euler angle in degrees
    }
    """
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Extract position and orientation values
        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        z = float(data.get('z', 0))
        rx_deg = float(data.get('rx', 0))
        ry_deg = float(data.get('ry', 0))
        rz_deg = float(data.get('rz', 0))

        # Convert Euler angles from degrees to radians for numpy trigonometric functions
        rx = np.radians(rx_deg)
        ry = np.radians(ry_deg)
        rz = np.radians(rz_deg)

        # --- Rotation Matrices ---
        # Rotation matrix around the X-axis
        R_x = np.array([[1, 0,          0],
                        [0, np.cos(rx), -np.sin(rx)],
                        [0, np.sin(rx), np.cos(rx)]])

        # Rotation matrix around the Y-axis
        R_y = np.array([[np.cos(ry),  0, np.sin(ry)],
                        [0,           1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])

        # Rotation matrix around the Z-axis
        R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                        [np.sin(rz), np.cos(rz),  0],
                        [0,          0,           1]])

        # Get the rotation convention, default to 'ZYX_extrinsic'
        convention = data.get('convention', 'ZYX_extrinsic')

        # Combined rotation matrix
        if convention == 'intrinsic_XYZ':
            R = R_x @ R_y @ R_z  # Intrinsic XYZ order
        else:
            # Default to ZYX extrinsic order (covers 'ZYX_extrinsic', 'extrinsic_XYZ', or unknown)
            R = R_z @ R_y @ R_x

        # --- Homogeneous Transformation Matrix ---
        # Create a 4x4 identity matrix
        transformation_matrix = np.identity(4)

        # Place the rotation matrix in the top-left 3x3 section
        transformation_matrix[:3, :3] = R

        # Place the translation vector in the last column
        transformation_matrix[:3, 3] = [x, y, z]

        # Convert the numpy array to a list of lists to be JSON serializable
        result = transformation_matrix.tolist()

        # Return the resulting matrix as a JSON response
        return jsonify(result)

    except (ValueError, TypeError) as e:
        # Handle cases where the input data is not valid
        return jsonify({"error": f"Invalid input data: {e}"}), 400
    except Exception as e:
        # Handle other potential errors
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000 in debug mode
    # The host '0.0.0.0' makes the server accessible from any IP address
    app.run(host='0.0.0.0', port=5000, debug=True)
