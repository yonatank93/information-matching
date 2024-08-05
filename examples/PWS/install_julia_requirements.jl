# Initialize Package Path
import Pkg

# Import registered packages
Pkg.add("PyCall")
Pkg.add(Pkg.PackageSpec(url="https://github.com/mktranstrum/NumDiffTools.jl"))

#Install Main Modeling Package
Pkg.add(Pkg.PackageSpec(url="https://git.physics.byu.edu/Modeling/Models.jl.git"))
