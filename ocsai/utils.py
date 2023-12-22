def can_render_md_html():
    try:
        from IPython import get_ipython

        # Check if IPython is running
        ipython = get_ipython()
        if ipython is None:
            return False  # Not in an IPython environment

        # Check if the environment is a Jupyter notebook
        # This is a heuristic check and may not be foolproof
        if 'zmqshell' in str(type(ipython)):
            return True
        else:
            return False

    except ImportError:
        # IPython is not installed
        return False
 

def mprint(*messages):
    '''If renderable, print as markdown; else print as text.'''
    full_message = ' '.join(str(message) for message in messages)
    renderable = can_render_md_html()
    if renderable:
        from IPython.display import Markdown, display
        display(Markdown(full_message))
    else:
        print(full_message)
