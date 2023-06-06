import os

def create_dir_tree(path, prefix=''):
    """Recursively print the directory tree starting at path."""
    print(prefix + os.path.basename(path) + '/')
    if os.path.isdir(path):
        files = os.listdir(path)
        for i, file in enumerate(files):
            if i == len(files) - 1:
                sub_prefix = prefix + '└── '
                next_prefix = prefix + '    '
            else:
                sub_prefix = prefix + '├── '
                next_prefix = prefix + '│   '
            create_dir_tree(os.path.join(path, file), prefix=next_prefix)

# Example usage
create_dir_tree('C:\Users\TurnerJ\Desktop\Git\LocalFunctionsProj\LocalFunctionProj\HttpExample\LocalFunction')