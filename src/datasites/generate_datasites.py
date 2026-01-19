import syft as sy

def generate_datasites(hospitals):
    """
    Generates a datasite for each hospital for the first time.
    It includes the following information
    """

    datasites = {}

    for hospital in hospitals:
        # Launch a new datasite for the hospital
        data_site = sy.orchestra.launch(name=hospital, reset=True)
        datasites[hospital] = data_site

    return datasites

