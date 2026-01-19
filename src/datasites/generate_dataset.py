import syft as sy

admin_email = "admin@tfg.unican.es"
admin_pwd = "admin"
main_contributor = sy.Contributor(name="Dataset Creator", role="Dataset Creator", email="test_dataset_creator@tfg.unican.es")

def generate_asset(name,desc,generated_data,mock,contributor):
    """
    Generates an asset for a dataset. 
    
    Parameters
    ----------
        name : str
            String that represents the name of the asset.
        desc : str
            String with the description of the asset.
        generated_data : primitives or pandas dataframe or numpy arrays
            Structure with the corresponding data.
        mock: primitives or pandas dataframe or numpy arrays
            Structure with fake data that has the same format as generated_data
            and can be used to test if data modifications on the datasite work properly.
        contributor: Contributor
            PySyft Contributor object that has created the asset.

    Returns
    -------
    result: Asset
        PySyft Asset with the corresponding parameters.
    """
    return sy.Asset(
        name=name,
        description=desc,
        data=generated_data,
        mock=mock,
        contributors=[contributor]
    )

def generate_dataset(name,desc,assets,contributor):
    """
    Generates a dataset for a datasite. 
    
    Parameters
    ----------
        name : str
            String that represents the name of the dataset.
        desc : str
            String with the description of the dataset.
        assets : List[Asset]
            A list containing the PySyft Assets that will be added to the dataset.
        contributor: Contributor
            PySyft Contributor object that has created the dataset.

    Returns
    -------
    result: Dataset
        PySyft Dataset with the corresponding parameters.
    """
    return sy.Dataset(
        name=name,
        description=desc,
        asset_list=assets,
        contributors=[contributor]
    )

def upload_data(datasites,data):
    for hospital, data_site in datasites.items():
        # Login to the datasite (for now as admin)
        admin_client = data_site.login(email="info@openmined.org", password="changethis")
        admin_client.account.set_email(admin_email)
        admin_client.account.set_password(admin_pwd, confirm=False)
        # Create an asset
        asset = generate_asset(
            name=f"Lung images of {hospital}",
            desc=f"An asset containing the images of {hospital}",
            generated_data=data[hospital],
            mock=sy.ActionObject.empty(), #Replace this later
            contributor=main_contributor
        )
        dataset = generate_dataset(
            name=f"{hospital} dataset",
            desc=f"A dataset containing the lung images of {hospital}",
            assets=[asset],
            contributor=main_contributor
        )
        # Later on I should instead make a training asset and a testing asset to upload to the datasite
        admin_client.upload_dataset(dataset)
        #### TESTING ####
        #################
        # We now try to retrieve the data. We use the admin user for now so we should be able
        # to see the data (a data scientist user wouldn't be able to do so).
        #retrieved_dataset = admin_client.datasets[0]
        #retrieved_asset = retrieved_dataset.assets[0]
        #private_data = retrieved_asset.data
        #print(private_data)
