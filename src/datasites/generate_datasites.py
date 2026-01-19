import syft as sy

def generate_datasites_orchestra_launch(hospitals):
    """
    Generates a datasite for each hospital for the first time.
    Each hospital is hosted in a different port in the same machine.
    Once the Python application stops executing, all datasites are removed.

    Parameters
    ----------
        hospitals : List[str]
            List containing the name of each hospital.

    Returns
    -------
        datasites: dict[str,sy.orchestra.datasite.DataSite]
            Dictionary that maps each hospital name to a PySyft Datasite.
    
    """

    datasites = {}

    for hospital in hospitals:
        # Launch a new datasite for the hospital
        data_site = sy.orchestra.launch(name=hospital, reset=True)
        datasites[hospital] = data_site

    return datasites

