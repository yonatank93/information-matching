module IEEE14_SEModel

using DelimitedFiles
using ComponentArrays
import Models

export model, x

WORK_DIR = @__DIR__

bus = readdlm(joinpath(WORK_DIR, "data//IEEE14_bus.txt"))
net = readdlm(joinpath(WORK_DIR, "data//IEEE14_net.txt"))

p = ComponentVector(;
    bus=ComponentVector(;
        # V= ones(size(bus,1)),
        # θ=zeros(size(bus,1))
        V=bus[:,6],
        θ=bus[:,7]
    ),
    net=ComponentVector(;
        fr=net[:,1],
        to=net[:,2],
        r =net[:,3],
        x =net[:,4],
        b =net[:,5],
		m =net[:,6]
    )
)

function x_guess(p)
	x = Vector{eltype(p)}(undef, 2length(p.bus.V))
	for i in eachindex(p.bus.V)
		x[2i-1:2i] .= p.bus.V[i], p.bus.θ[i]
	end
	return x
end

# # branch power flows
# Pij(Vi, θi, Vj, θj, Y, ϕ      ) = Y*Vi*(Vj*cos(θi - θj + ϕ)   - Vi*cos(ϕ)    )          # entering i from j
# Qij(Vi, θi, Vj, θj, Y, ϕ, B   ) = Y*Vi*(Vj*sin(θi - θj + ϕ)   - Vi*sin(ϕ)    ) + B*Vi^2 # entering i from j
# Pfr(Vf, θf, Vt, θt, Y, ϕ,    m) = Y*Vf*(Vt*cos(θf - θt + ϕ)/m - Vf*cos(ϕ)/m^2)          # entering f from t
# Qfr(Vf, θf, Vt, θt, Y, ϕ, B, m) = Y*Vf*(Vt*sin(θf - θt + ϕ)/m - Vf*sin(ϕ)/m^2) + B*Vf^2 # entering f from t
# Pto(Vt, θt, Vf, θf, Y, ϕ,    m) = Y*Vt*(Vf*cos(θt - θf + ϕ)/m - Vt*cos(ϕ)    )          # entering t from f
# Qto(Vt, θt, Vf, θf, Y, ϕ,    m) = Y*Vt*(Vf*sin(θt - θf + ϕ)/m - Vt*sin(ϕ)    )          # entering t from f

# branch currents
Idji(Vi, θi, Vj, θj, Y, ϕ, B   ) = Y*(Vj*cos(θj - ϕ)   - Vi*cos(θi - ϕ)    ) + B*Vi*sin(θi) # entering i from j
Iqji(Vi, θi, Vj, θj, Y, ϕ, B   ) = Y*(Vj*sin(θj - ϕ)   - Vi*sin(θi - ϕ)    ) - B*Vi*cos(θi) # entering i from j
Idfr(Vf, θf, Vt, θt, Y, ϕ, B, m) = Y*(Vt*cos(θt - ϕ)/m - Vf*cos(θf - ϕ)/m^2) + B*Vf*sin(θf) # entering f from t
Iqfr(Vf, θf, Vt, θt, Y, ϕ, B, m) = Y*(Vt*sin(θt - ϕ)/m - Vf*sin(θf - ϕ)/m^2) - B*Vf*cos(θf) # entering f from t
Idto(Vt, θt, Vf, θf, Y, ϕ,    m) = Y*(Vf*cos(θf - ϕ)/m - Vt*cos(θt - ϕ)    )                # entering t from f
Iqto(Vt, θt, Vf, θf, Y, ϕ,    m) = Y*(Vf*sin(θf - ϕ)/m - Vt*sin(θt - ϕ)    )                # entering t from f

function obs!(y, x, p)
	# in/x: Vi, θi
	# out/y: Vi, θi, Idji, Iqji, Idi, Iqi

	# eqs:
	# Vi = Vi
	# θi = θi
	n = 2length(p.bus.V)
	y[1:n] .= @view(x[1:n])

	# i_ji = y_ji*(v_j - v_i) - im*b_i*v_i (from j to i)
	# i_fr = y_ji*(v_j*cis(φ)/m - v_i/m^2) - im*b_i*v_i (from j to i)
	# i_to = y_ji*(v_i/m - v_j*cis(φ)) (from i to j)

	# i_ji = Y_ji*(V_j*cis(θ_j - ϕ) - V_i*cis(θ_i - ϕ)) - im*b_i*V_i*cis(θ_i) (from j to i)
	# i_fr = Y_ji*(V_j*cis(θ_j - ϕ + φ)/m - V_i*cis(θ_i - ϕ)/m^2) - im*b_i*V_i*cis(θ_i) (from j to i)
	# i_to = Y_ji*(V_i*cis(θ_i - ϕ)/m - V_j*cis(θ_j - ϕ + φ)) (from i to j)

	# Id_ji = Y_ji*(V_j*cos(θ_j - ϕ) - V_i*cos(θ_i - ϕ)) + b_i*V_i*sin(θ_i) (from j to i)
	# Iq_ji = Y_ji*(V_j*sin(θ_j - ϕ) - V_i*sin(θ_i - ϕ)) - b_i*V_i*cos(θ_i) (from j to i)
	# Id_fr = Y_ji*(V_j*cos(θ_j - ϕ + φ)/m - V_i*cos(θ_i - ϕ)/m^2) + b_i*V_i*sin(θ_i) (from j to i)
	# Iq_fr = Y_ji*(V_j*sin(θ_j - ϕ + φ)/m - V_i*sin(θ_i - ϕ)/m^2) - b_i*V_i*cos(θ_i) (from j to i)
	# Id_to = Y_ji*(V_i*cos(θ_i - ϕ)/m - V_j*cos(θ_j - ϕ + φ)) (from i to j)
	# Iq_to = Y_ji*(V_i*sin(θ_i - ϕ)/m - V_j*sin(θ_j - ϕ + φ)) (from i to j)

	# Idi = -∑j Idji (entering i from j)
	# Iqi = -∑j Iqji (entering i from j)
	# 0 = -∑j Idji (zero injection bus)
	# 0 = -∑j Iqji (zero injection bus)
	# start from 0 for updating assignments below:
	y[n+1:2n] .= 0
	for i in eachindex(p.net.to)
		fr = Int(p.net.fr[i])
		to = Int(p.net.to[i])
		Vf = x[2fr-1]
		θf = x[2fr  ]
		Vt = x[2to-1]
		θt = x[2to  ]

		R = p.net.r[i]
		X = p.net.x[i]
		B = p.net.b[i]
		m = p.net.m[i]
		Y = 1/√(R^2 + X^2)
		ϕ = atan(X, R)

		if m == 0 # lines
			y[n+2fr-1] -= y[2n+4i-3] = Idji(Vf, θf, Vt, θt, Y, ϕ, B)
			y[n+2fr  ] -= y[2n+4i-2] = Iqji(Vf, θf, Vt, θt, Y, ϕ, B)
			y[n+2to-1] -= y[2n+4i-1] = Idji(Vt, θt, Vf, θf, Y, ϕ, B)
			y[n+2to  ] -= y[2n+4i  ] = Iqji(Vt, θt, Vf, θf, Y, ϕ, B)
		else # transformers
			y[n+2fr-1] -= y[2n+4i-3] = Idfr(Vf, θf, Vt, θt, Y, ϕ, B, m)
			y[n+2fr  ] -= y[2n+4i-2] = Iqfr(Vf, θf, Vt, θt, Y, ϕ, B, m)
			y[n+2to-1] -= y[2n+4i-1] = Idto(Vt, θt, Vf, θf, Y, ϕ,    m)
			y[n+2to  ] -= y[2n+4i  ] = Iqto(Vt, θt, Vf, θf, Y, ϕ,    m)
		end
	end

	nothing
end

function update_guess!(p, x)
	for i in eachindex(p.bus.V)
		p.bus.V[i], p.bus.θ[i] = @view(x[2i-1:2i])
	end
	nothing
end

function h(x; bus=bus, net=net, PMU_idx=1:size(bus,1))
	n = size(bus,1)
	p = ComponentVector{promote_type(
			eltype(x),
			eltype(net),
		)}(;
		bus=ComponentVector(;
			# V= ones(n),
			# θ=zeros(n)
			V=bus[:,6],
			θ=bus[:,7]
		),
		net=ComponentVector(;
			fr=net[:,1],
			to=net[:,2],
			r =net[:,3],
			x =net[:,4],
			b =net[:,5],
			m =net[:,6]
		)
	)
	for i in eachindex(p.bus.V)
		p.bus.V[i], p.bus.θ[i] = @view(x[2i-1:2i])
	end
	y = Array{promote_type(eltype(x), eltype(p))}(undef, 4n + 4length(p.net.to))
	obs!(y, x, p)

	idx = Int[]
	# A PMU at Bus i gives measurements for Vi, θi, Idji ∀ j, and Iqji ∀ j
	for i in PMU_idx
		append!(idx,    2i-1:2i)    # Vi, θi
		# append!(idx, 2n+2i-1:2n+2i) # Pi, Qi
		for k in findall(==(i), p.net.fr)
			append!(idx, 4n+4k-3:4n+4k-2) # Idji, Iqji
		end
		for k in findall(==(i), p.net.to)
			append!(idx, 4n+4k-1:4n+4k)   # Idji, Iqji
		end
	end
	# Zero injection bus gives "measurements" for Idi and Iqi (i.e., 0), regardless of PMU placement
	for i in findall(==(1), bus[:,1])     # PQ buses (no gen)
		if bus[i,2] == 0 && bus[i,3] == 0 # where the load is 0
			append!(idx, 2n+2i-1:2n+2i) # Idi, Iqi
		end
	end
	sort!(idx)
	unique!(idx)

	return y[idx]
end

x0 = x_guess(p)

y = h(x0)

model = Models.Model(length(y), length(x0), x->h(x) - y, Val(true), "IEEE14 Model")

end # module
