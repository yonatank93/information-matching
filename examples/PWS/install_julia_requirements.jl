# Initialize Package Path
import Pkg

# Import registered packages
Pkg.add("PyCall")
Pkg.add(Pkg.PackageSpec(url="https://github.com/mktranstrum/NumDiffTools.jl"))
