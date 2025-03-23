#!/usr/bin/env python3
"""
Script to generate the drift dashboard template file.
This is a workaround for the gitignore restrictions.
"""
import os
from pathlib import Path

# Define the template directory
TEMPLATE_DIR = Path(__file__).parent.parent / "app" / "templates" / "pages" / "researcher"

# Create the directory if it doesn't exist
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Define the template content
TEMPLATE_CONTENT = """{% extends "base.html" %}

{% block title %}Drift Detection Dashboard{% endblock %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-12">
            <!-- The drift dashboard will be dynamically created by JavaScript -->
            <div id="driftDashboardContainer" class="drift-dashboard">
                <!-- Content will be inserted here by drift_dashboard_init.js -->
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
    <!-- Load Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
    
    <!-- Load our dashboard initialization script -->
    <script src="{{ url_for('static', path='js/drift_dashboard_init.js') }}"></script>
    
    {% if extra_js %}
        {% for js in extra_js %}
            <script src="{{ js }}"></script>
        {% endfor %}
    {% endif %}
{% endblock %}
"""

# Write the template file
template_path = TEMPLATE_DIR / "drift_dashboard.html"
with open(template_path, "w") as f:
    f.write(TEMPLATE_CONTENT)

print(f"Template file created at: {template_path}")
