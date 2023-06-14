from collections import namedtuple
from typing import Any, Dict, Sequence, Union

from aot_compile.static_analysis import JITStub
from dataclasses import dataclass

from triton.compiler.code_generator import kernel_suffix

instance_descriptor = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"])


def _exists(v):
    return v is not None


def _valid_triton_ty_ann(ann) -> Union[str, None]:
    """
    Takes in a string for type annotation. If string is valid return the type (without pointer prefix)
    """
    if ann[0] == "*":
        ann = ann[1:]
    tys = {
        "fp8": "triton.language.float8",
        "fp16": "triton.language.float16",
        "bf16": "triton.language.bfloat16",
        "fp32": "triton.language.float32",
        "fp64": "triton.language.float64",
        "i1": "triton.language.int1",
        "i8": "triton.language.int8",
        "i16": "triton.language.int16",
        "i32": "triton.language.int32",
        "i64": "triton.language.int64",
        "u8": "triton.language.uint8",
        "u16": "triton.language.uint16",
        "u32": "triton.language.uint32",
        "u64": "triton.language.uint64",
        "B": "triton.language.int1",
    }
    if ann in tys:
        return ann
    return


@dataclass
class CompileMetadata:
    arg_names: Sequence[str]
    """ Names of input arguments """
    signature: Dict[int, str]
    """ Triton type annotations of function argument """
    constants: Dict[str, Union[int, JITStub]]
    specializations: instance_descriptor
    compiled_function_name: str
    """ kernel name as generetaed by compiler """
    docstr: str
    """ Represents argument types and constant values"""


def parse_signature(type_anns: Sequence[str]) -> Dict[int, str]:
    signature = {}
    for argnum, type_ann in enumerate(type_anns):
        assert _exists(
            _valid_triton_ty_ann(type_ann)
        ), f"Bad type annotaoin {type_ann} is not valid"
        signature[argnum] = type_ann.strip()
    return signature


def parse_specializations(spec_ann: Sequence[str]) -> instance_descriptor:

    div_16 = set()
    is_1 = set()

    for argnum, spec in enumerate(spec_ann):
        try:
            spec_val = int(spec)
            if spec_val == 16:
                div_16.add(argnum)
            elif spec_val == 1:
                is_1.add(argnum)
        except ValueError:
            # No specializations
            continue

    return instance_descriptor(divisible_by_16=div_16, equal_to_1=is_1)


def _valid_constant(const: str, const_name: str, global_scope: Dict[str, Any]) -> Union[int, float, JITStub]:
    if const.isnumeric():
        return int(const)

    try:
        return float(const)
    except ValueError:
        # not a float, so assume string
        pass

    try:
        return global_scope[const]
    except KeyError:
        from argparse import ArgumentError

        raise ArgumentError(f"[Bad Value: {const_name}]{const} is not a valid number and no global object with this name exists")


def compilation_metadata_from_args(
    kernel: JITStub, kernel_args: Sequence[str]
) -> CompileMetadata:
    from argparse import ArgumentParser

    parser = ArgumentParser(description=f"Meta params for {kernel.__name__}")
    arg_names = kernel.arg_names
    consts = []
    ker_arg_names = []

    for arg_num, arg in enumerate(arg_names):
        if arg_num in kernel.constants:
            consts.append(arg)
        else:
            ker_arg_names.append(arg)

    sig = ",".join(ker_arg_names)
    parser.add_argument(
        "--signature",
        "-s",
        nargs=len(ker_arg_names),
        type=str,
        required=True,
        metavar=tuple(ker_arg_names),
        help=f"Provide annotations for the following (in order) {sig} e.g. *fp32:16, i32:1",
    )

    # TODO: add support for defaults
    for cname in consts:
        parser.add_argument(
            f"--{cname}",
            type=lambda x: _valid_constant(x, cname, kernel.__globals__),
            help="Constant value for kernel compilation",
            required=True,
        )

    args = parser.parse_args(args=kernel_args)

    signature = []
    specializations = []

    doc_str = []

    for argnum, arg_str in enumerate(args.signature):
        type_ann, spec = arg_str.split(":")
        doc_str.append(f"{ker_arg_names[argnum]}: {arg_str}")
        signature.append(type_ann)
        specializations.append(spec)

    arg_docstr = ",".join(doc_str)

    const_dict = vars(args)
    const_dict.pop("signature")
    conts_docstr = ",".join([f"{k}: {v}" for k, v in const_dict.items()])

    specials = parse_specializations(specializations)
    function_name = f"{kernel.__name__}_{kernel_suffix(signature=ker_arg_names, specialization=specials)}"

    docstr_parts = [kernel.__name__, "---", arg_docstr, conts_docstr]

    if kernel.__doc__ is not None:
        docstr_parts += kernel.__doc__.split("\n")

    docstr = "\n\t".join(docstr_parts)

    return CompileMetadata(
        arg_names=ker_arg_names,
        signature=parse_signature(signature),
        constants=const_dict,
        specializations=specials,
        compiled_function_name=function_name,
        docstr=docstr,
    )
