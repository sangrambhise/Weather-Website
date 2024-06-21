markdown
Copy code
# Weatherpedia

Weatherpedia is a Django web application that provides weather forecasts and analysis for different cities.

## Features

- **Home**: Displays the home page with basic information.
- **Forecast**: Provides weather forecasts for various cities.
- **About Us**: Information about the project and its contributors.
- **Hot and Cold Cities**: Shows the hottest and coldest cities.

## Screenshots

![Screenshot (1757)](https://github.com/sangrambhise/Weather-Website/assets/114818287/7960d49b-51e4-45db-b1ff-4035b5f0da9b), ![Screenshot (1758)](https://github.com/sangrambhise/Weather-Website/assets/114818287/cfeff8eb-797c-4089-98c8-7fdb3c9d28fe), ![Screenshot (1759)](https://github.com/sangrambhise/Weather-Website/assets/114818287/2a8d6532-45b8-4193-8511-7b2c86aab2bd)., ![Screenshot (1760)](https://github.com/sangrambhise/Weather-Website/assets/114818287/cee6870d-119d-4fd3-bfd8-a239235e53a1), ![Screenshot (1761)](https://github.com/sangrambhise/Weather-Website/assets/114818287/7eefb0d0-f4c0-4e12-9176-da75be17dc2d)

 ## Installation

To run this project locally:

1. Clone the repository:
   git clone https://github.com/sangrambhise/weather website.git
   cd weather

2. Install dependencies:
pip install -r requirements.txt

3. Apply database migrations:
python manage.py migrate

4. Load initial data (if any):
python manage.py loaddata <kanpur.csv>

5. Replace API Key:
Enter you OpenWeatherMap API Key in "api_key" in views.py and in front of "app_id=" in index.py

5. Run the development server:
python manage.py runserver

6. Open your browser and go to http://127.0.0.1:8000/ to see the web application.

## Usage
Navigate to http://127.0.0.1:8000/ to access the Weatherpedia application.
Explore different pages like Home, Forecast, About Us, and Hot and Cold Cities.
Home Page: Navigate to the home page to get an overview of the project.
Prediction Page: Go to the prediction page to view weather predictions using Random Forest.
Top Cities Page: View the top hottest and coldest cities using the navigation.

## Technologies Used
Django
HTML/CSS
Bootstrap

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the project.
2. Create your feature branch (git checkout -b feature/AmazingFeature).
3. Commit your changes (git commit -m 'Add some AmazingFeature').
4. Push to the branch (git push origin feature/AmazingFeature).
5. Open a pull request.

## Project Structure
weather_project/: This is the root directory of your Django project.

weather_app/: This is your Django app directory.

migrations/: Contains database migration files.

static/: Directory for static files like CSS, JavaScript, and images.

css/: CSS files.

js/: JavaScript files.

img/: Image files.

templates/: Directory for HTML templates.

weather_app/: App-specific templates.

__init__.py: Initializes the app as a Python module.

admin.py: Django admin configuration.

apps.py: App configuration.

models.py: Django models for the app.

tests.py: Test cases for the app.

urls.py: URL configuration for the app.

views.py: Views (controllers) for the app.

manage.py: Django's command-line utility for administrative tasks.

README.md: Markdown file containing project information.

requirements.txt: List of Python packages required by the project.

.gitignore: Git ignore file to exclude certain files and directories from version control.

## Contact
Project Link: https://github.com/your-username/your-repository

## License
This project is licensed under the MIT License - see the LICENSE file for details.
