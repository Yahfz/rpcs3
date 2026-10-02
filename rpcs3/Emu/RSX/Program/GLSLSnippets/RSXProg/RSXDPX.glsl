R"(
// 1 = +Inf, 2 = -Inf, 4 = NaN, both Inf signs also give NaN
uint _dpx_product_class(uint bits_a, uint bits_b)
{
	uint magnitude_a = bits_a & 0x7fffffffu;
	uint magnitude_b = bits_b & 0x7fffffffu;
	bool nan = magnitude_a > 0x7f800000u || magnitude_b > 0x7f800000u;
	bool inf_a = magnitude_a == 0x7f800000u;
	bool inf_b = magnitude_b == 0x7f800000u;
	bool zero_a = (magnitude_a & 0x7f800000u) == 0u;
	bool zero_b = (magnitude_b & 0x7f800000u) == 0u;
	uint infinity = ((bits_a ^ bits_b) & 0x80000000u) != 0u ? 2u : 1u;
	return (nan || (inf_a && zero_b) || (inf_b && zero_a)) ? 4u : ((inf_a || inf_b) ? infinity : 0u);
}

float _dpx_dot(uvec4 bits_a, uvec4 bits_b, uint product_lanes, uint direct_bits, bool vertex)
{
	uint value_class = _dpx_product_class(0x3f800000u, direct_bits);
	int direct_exponent = int((direct_bits >> 23) & 0xffu);
	int direct_term = int((0x800000u | (direct_bits & 0x7fffffu)) << 1);
	direct_term = direct_exponent == 0 ? 0 : (direct_bits >= 0x80000000u ? -direct_term : direct_term);
	ivec4 magnitude = ivec4(0, 0, 0, direct_term);
	ivec4 order = ivec4(-1000000, -1000000, -1000000, direct_term == 0 ? -1000000 : 2 * (direct_exponent - 126));
	uint cutoff = vertex ? 19u : 17u;
	uint correction = vertex ? 0x780000u : 0x520000u;

	for (uint lane = 0u; lane < product_lanes; ++lane)
	{
		uint word_a = bits_a[lane];
		uint word_b = bits_b[lane];
		value_class |= _dpx_product_class(word_a, word_b);

		uint significand_a = 0x800000u | (word_a & 0x7fffffu);
		uint significand_b = 0x800000u | (word_b & 0x7fffffu);
		uint raw_high;
		uint raw_low;
		umulExtended(significand_a, significand_b, raw_high, raw_low);

		uint omitted = 0u;
		for (uint bit = 0u; bit < cutoff; ++bit)
		{
			if ((significand_b & (1u << bit)) != 0u)
			{
				uint width = cutoff - bit;
				uint mask = (1u << width) - 1u;
				omitted += (significand_a & mask) << bit;
			}
		}

		uint subtracted_low = raw_low - omitted;
		uint subtracted_high = raw_high - uint(raw_low < omitted);
		uint approximate_low = subtracted_low + correction;
		uint approximate_high = subtracted_high + uint(approximate_low < subtracted_low);
		int width = (approximate_high & 0x8000u) != 0u ? 48 : 47;
		int serialization_bits = (vertex && (raw_high & 0x8000u) != 0u) ? 25 : 24;
		int discarded_bits = width - serialization_bits;
		uint serialized = (approximate_high << (32 - discarded_bits)) | (approximate_low >> discarded_bits);
		int leading = int((word_a >> 23) & 0xffu) + int((word_b >> 23) & 0xffu) - 300 + width;
		bool negative = ((word_a ^ word_b) & 0x80000000u) != 0u;
		bool present = (word_a & 0x7f800000u) != 0u && (word_b & 0x7f800000u) != 0u;

		// overflow happens before the sum, so opposite Inf give NaN
		value_class |= (present && leading > 128) ? (negative ? 2u : 1u) : 0u;
		// products flush before the sum, even when their sum would be normal
		present = present && leading > -126;
		int term = int(serialized << (25 - serialization_bits));
		magnitude[lane] = present ? (negative ? -term : term) : 0;
		order[lane] = present ? 2 * leading + width - 47 : -1000000;
	}

	// align fragment terms to 26 bits and vertex terms to 29 bits
	// vertices use 28 bits if any term with the largest exponent is 47 bits wide
	int max_leading = max(max(order.x, order.y), max(order.z, order.w)) >> 1;
	int grid_bits = vertex ? (any(equal(order, ivec4(2 * max_leading))) ? 28 : 29) : 26;
	int step_exponent = max_leading - grid_bits;
	// shift left by 4 first so alignment only needs right shifts
	uvec4 shift = uvec4(clamp(ivec4(step_exponent + 29) - (order >> 1), 0, 31));
	ivec4 units = ivec4((uvec4(abs(magnitude)) << 4u) >> shift) * sign(magnitude);
	int sum = units.x + units.y + units.z + units.w;

	uint sum_magnitude = uint(abs(sum));
	int leading_bit = findMSB(sum_magnitude);
	int binary_exponent = step_exponent + leading_bit;
	uint sign = sum < 0 ? 0x80000000u : 0u;
	uint significand = leading_bit >= 23 ?
		sum_magnitude >> (leading_bit - 23) : sum_magnitude << (23 - leading_bit);
	uint result = sign | (uint(binary_exponent + 127) << 23) | (significand & 0x7fffffu);
	result = binary_exponent > 127 ? sign | 0x7f800000u : result;
	result = (sum == 0 || binary_exponent < -126) ? 0u : result;
	// RSX returns this NaN payload for vertex and fragment dot products
	uint special = value_class >= 3u ? 0x7fffffffu : (value_class == 2u ? 0xff800000u : 0x7f800000u);
	return uintBitsToFloat(value_class != 0u ? special : result);
}

float _fp_dp2(vec4 a, vec4 b)
{
	return _dpx_dot(floatBitsToUint(a), floatBitsToUint(b), 2u, 0u, false);
}

float _fp_dp2a(vec4 a, vec4 b, float c)
{
	return _dpx_dot(floatBitsToUint(a), floatBitsToUint(b), 2u, floatBitsToUint(c), false);
}

float _fp_dp3(vec4 a, vec4 b)
{
	return _dpx_dot(floatBitsToUint(a), floatBitsToUint(b), 3u, 0u, false);
}

float _fp_dp4(vec4 a, vec4 b)
{
	return _dpx_dot(floatBitsToUint(a), floatBitsToUint(b), 4u, 0u, false);
}

float _vp_dp3(vec4 a, vec4 b)
{
	return _dpx_dot(floatBitsToUint(a), floatBitsToUint(b), 3u, 0u, true);
}

float _vp_dph(vec4 a, vec4 b)
{
	return _dpx_dot(floatBitsToUint(a), floatBitsToUint(b), 3u, floatBitsToUint(b.w), true);
}

float _vp_dp4(vec4 a, vec4 b)
{
	return _dpx_dot(floatBitsToUint(a), floatBitsToUint(b), 4u, 0u, true);
}
)"
