import os

def set_project(project_name):
    # Define the path for the "projects" directory
    projects_dir = os.path.join(os.getcwd(), "projects")
    
    # Check if the "projects" directory exists, if not create it
    if not os.path.exists(projects_dir):
        os.makedirs(projects_dir)
    
    # Define the path for the specific project directory within "projects"
    project_path = os.path.join(projects_dir, project_name)
    
    # Check if the project directory exists, if not create it
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    
    # Return the path to the project directory
    return project_path

# Example usage:
# project_path = set_project("MyNewProject")
# print("Project path:", project_path)
