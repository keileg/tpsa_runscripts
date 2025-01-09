# This file was used to create a Docker image that can be used to reproduce results
# obtained with the two-point stress approximation method.
#
# The runscripts are available at https://github.com/keileg/tpsa_runscripts.git.
#  
# The runscripts are set up as an application of PorePy, and the Dockerfile is based on
# the PorePy development image. An the time of creation, the PorePy development image
# does not contain all functionality needed to run the TPSA runscripts, and the
# Dockerfile therefore sets up PorePy to run in a separate branch.

# Start from a base image with PorePy installed.
FROM porepy/dev:latest

# Install PyAMG.
RUN pip install pyamg

# Enter the porepy directory.
WORKDIR /workdir/porepy

# Checkout the branch with the TPSA implementation.
RUN git remote add tpsa_repo https://github.com/keileg/porepy.git
RUN git fetch tpsa_repo 
RUN git switch -c tpsa tpsa_repo/tpsa_no_cosserat

# Install the Tpsa runscripts.
WORKDIR /workdir/
RUN git clone https://github.com/keileg/tpsa_runscripts.git

WORKDIR /workdir/tpsa_runscripts


# Set an entrypoint to the directory with the runscript.
ENTRYPOINT ["bash"]