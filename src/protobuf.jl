# A reader of the protobuf wire format, enough for the Core ML files in which Kraken saves its models. A message is a
# view of the bytes of the file; its fields are found by number, as Core ML's .proto files number them.

const Bytes = SubArray{UInt8, 1, Vector{UInt8}, Tuple{UnitRange{Int}}, true}

"""
	Field

A field of a protobuf message: its number, its wire type (0 varint, 1 fixed 64 bits, 2 length-delimited, 5 fixed
32 bits), its value for wire types 0, 1 and 5, and its bytes for wire type 2.
"""
struct Field
	number::Int
	wire::Int
	value::UInt64
	bytes::Bytes
end

"""The varint that starts at byte `i` of `bytes`, and the index of the byte after it."""
function varint(bytes::AbstractVector{UInt8}, i::Int)
	x = UInt64(0)
	shift = 0
	while true
		b = bytes[i]
		i += 1
		x |= UInt64(b & 0x7f) << shift
		b < 0x80 && return x, i
		shift += 7
	end
end

"""The fields of the message `message`, in the order of the file."""
function fields(message::Bytes)
	parent, offset = message.parent, first(message.indices[1]) - 1
	found = Field[]
	i, stop = 1, length(message)
	empty = view(parent, 1:0)
	while i <= stop
		key, i = varint(message, i)
		number, wire = Int(key >> 3), Int(key & 0x07)
		if wire == 0
			value, i = varint(message, i)
			push!(found, Field(number, wire, value, empty))
		elseif wire == 1
			push!(found, Field(number, wire, ltoh(reinterpret(UInt64, message[i:i + 7])[1]), empty))
			i += 8
		elseif wire == 2
			len, i = varint(message, i)
			push!(found, Field(number, wire, UInt64(0), view(parent, offset + i:offset + i + Int(len) - 1)))
			i += Int(len)
		elseif wire == 5
			push!(found, Field(number, wire, UInt64(ltoh(reinterpret(UInt32, message[i:i + 3])[1])), empty))
			i += 4
		else
			error("Wire type $(wire) of field $(number) is not one a Core ML model uses.")
		end
	end
	found
end

"""The submessages of field `number` of `message`."""
messages(message::Bytes, number::Int) = [f.bytes for f in fields(message) if f.number == number]

"""The submessage of field `number` of `message`, `nothing` when it is absent."""
function message(message::Bytes, number::Int)
	found = messages(message, number)
	isempty(found) ? nothing : first(found)
end

"""The integer of field `number` of `message`, signed as protobuf's int32 and int64 are, or `default` when it is absent."""
function integer(message::Bytes, number::Int, default::Int = 0)
	for f in Iterators.reverse(fields(message))
		f.number == number && return Int(reinterpret(Int64, f.value))
	end
	default
end

"""The integers of the repeated field `number` of `message`, packed or not."""
function integers(message::Bytes, number::Int)
	found = Int[]
	for f in fields(message)
		f.number == number || continue
		if f.wire == 2
			i = 1
			while i <= length(f.bytes)
				x, i = varint(f.bytes, i)
				push!(found, Int(reinterpret(Int64, x)))
			end
		else
			push!(found, Int(reinterpret(Int64, f.value)))
		end
	end
	found
end

"""The string of field `number` of `message`, or `default` when it is absent."""
function text(message::Bytes, number::Int, default::String = "")
	found = messages(message, number)
	isempty(found) ? default : String(copy(last(found)))
end

"""The numbers of the repeated float field `number` of `message`, packed or not."""
function floats(message::Bytes, number::Int)
	found = Float32[]
	for f in fields(message)
		f.number == number || continue
		if f.wire == 2
			append!(found, ltoh.(reinterpret(Float32, copy(f.bytes))))
		else
			push!(found, reinterpret(Float32, UInt32(f.value)))
		end
	end
	found
end

"""The double of field `number` of `message`, or `default` when it is absent."""
function double(message::Bytes, number::Int, default::Float64 = 0.0)
	for f in Iterators.reverse(fields(message))
		f.number == number && return reinterpret(Float64, f.value)
	end
	default
end
