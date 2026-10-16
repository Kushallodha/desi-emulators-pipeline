(input, output, D, Pkemu) -> begin
	scale = exp(input[2]) * 1e-10 * D^2
	n = length(output) ÷ 6
	vcat(output[1:(3n)] .* scale, output[(3n+1):(6n)] .* scale^3)
end