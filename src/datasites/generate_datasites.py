import syft as sy

admin_email = "admin@tfg.unican.es"
admin_pwd = "admin"

def generate_datasites(hospitals):
    """
    Generates a datasite for each hospital for the first time.
    It includes the following information
    """

    for hospital in hospitals:
        # Launch a new datasite for the hospital
        data_site = sy.orchestra.launch(name=hospital, reset=True)
        # Log in as admin
        admin_client = data_site.login(email="info@openmined.org",password="changethis")
        admin_client.account.set_email(admin_email)
        admin_client.account.set_password(admin_pwd, confirm=False)

        #admin_client.users