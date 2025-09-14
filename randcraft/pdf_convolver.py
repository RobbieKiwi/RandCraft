from collections.abc import Sequence

from scipy.stats import norm

from randcraft.models import AlgebraicFunction
from randcraft.pdfs.continuous import ContinuousDistributionFunction, ScaledDistributionFunction
from randcraft.pdfs.discrete import DiracDeltaDistributionFunction, DiscreteDistributionFunction
from randcraft.pdfs.multi import MultiDistributionFunction
from randcraft.pdfs.scipy_pdf import ScipyDistributionFunction


class PdfConvolver:
    @classmethod
    def convolve_pdfs(
        cls, pdfs: Sequence[DiscreteDistributionFunction | ContinuousDistributionFunction]
    ) -> DiscreteDistributionFunction | ContinuousDistributionFunction:
        assert len(pdfs) >= 2, "At least two PDFs are required for combination."
        if not all(isinstance(pdf, ContinuousDistributionFunction | DiscreteDistributionFunction) for pdf in pdfs):
            types = {pdf.__class__.__name__ for pdf in pdfs}
            raise TypeError(
                f"All PDFs must be instances of {ContinuousDistributionFunction | DiscreteDistributionFunction}, got: {types}"
            )

        if all(rv.short_name == "normal" for rv in pdfs):
            return cls.convolve_normals(pdfs=pdfs)  # type: ignore

        continuous_pdfs, discrete_pdfs = cls.unpack_pdfs(pdfs=pdfs)

        if len(discrete_pdfs) >= 2:
            # First we convolve together all discrete pdfs
            reduced_discrete = cls.convolve_discretes(pdfs=discrete_pdfs)

            if not len(continuous_pdfs):
                return reduced_discrete

            discrete_pdfs = [reduced_discrete]

        discrete_pdf = discrete_pdfs[0] if len(discrete_pdfs) else None
        if discrete_pdf is not None:
            if len(continuous_pdfs) == 1 and isinstance(discrete_pdf, DiracDeltaDistributionFunction):
                return continuous_pdfs[0].add_constant(discrete_pdf.value)
            return MultiDistributionFunction(continuous_pdfs=continuous_pdfs, discrete_pdf=discrete_pdf)

        if len(continuous_pdfs) == 1:
            return continuous_pdfs[0]
        return MultiDistributionFunction(continuous_pdfs=continuous_pdfs)

    @classmethod
    def unpack_pdfs(
        cls, pdfs: Sequence[DiscreteDistributionFunction | ContinuousDistributionFunction]
    ) -> tuple[list[ContinuousDistributionFunction], list[DiscreteDistributionFunction]]:
        continuous_pdfs: list[ContinuousDistributionFunction] = []
        discrete_pdfs: list[DiscreteDistributionFunction] = []

        for pdf in pdfs:
            if isinstance(pdf, ScaledDistributionFunction):
                af = pdf.algebraic_function
                no_offset_continuous = ScaledDistributionFunction(
                    inner=pdf.inner, algebraic_function=AlgebraicFunction(scale=af.scale, offset=0.0)
                )
                continuous_pdfs.append(no_offset_continuous)
                if af.offset != 0.0:
                    offset_distribution = DiracDeltaDistributionFunction(value=af.offset)
                    discrete_pdfs.append(offset_distribution)
            elif isinstance(pdf, MultiDistributionFunction):
                inner_continuous, inner_discrete = cls.unpack_pdfs(pdf.pdfs)
                continuous_pdfs.extend(inner_continuous)
                discrete_pdfs.extend(inner_discrete)
            elif isinstance(pdf, ContinuousDistributionFunction):
                continuous_pdfs.append(pdf)
            elif isinstance(pdf, DiscreteDistributionFunction):
                discrete_pdfs.append(pdf)
            else:
                raise TypeError(f"Unsupported PDF type: {type(pdf)}")

        return continuous_pdfs, discrete_pdfs

    @classmethod
    def convolve_discretes(cls, pdfs: list[DiscreteDistributionFunction]) -> DiscreteDistributionFunction:
        assert all(isinstance(pdf, DiscreteDistributionFunction) for pdf in pdfs)

        def to_dict(x: DiscreteDistributionFunction) -> dict[float, float]:
            return {k: v for k, v in zip(x.values, x.probabilities)}

        def from_dict(x: dict[float, float]) -> DiscreteDistributionFunction:
            values = list(x.keys())
            probabilities = list(x.values())
            return DiscreteDistributionFunction(values=values, probabilities=probabilities)

        def convolve_two(x: dict[float, float], y: dict[float, float]) -> dict[float, float]:
            output_dict: dict[float, float] = {}
            for x1, p1 in x.items():
                for x2, p2 in y.items():
                    x3 = x1 + x2
                    p3 = p1 * p2
                    output_dict[x3] = output_dict.get(x3, 0.0) + p3
            return output_dict

        def convolve_iteratively(dicts: list[dict[float, float]]) -> dict[float, float]:
            if len(dicts) == 1:
                return dicts[0]
            first_two_joined = convolve_two(dicts[0], dicts[1])
            others = dicts[2:]
            if not len(others):
                return first_two_joined
            new_dicts = [first_two_joined, *others]
            return convolve_iteratively(new_dicts)

        dict_list = [to_dict(pdf) for pdf in pdfs]
        result_dict = convolve_iteratively(dict_list)
        result_pdf = from_dict(result_dict)
        if len(result_pdf.values) == 1:
            return DiracDeltaDistributionFunction(value=result_pdf.values[0])
        return result_pdf

    @classmethod
    def convolve_normals(cls, pdfs: list[ScipyDistributionFunction]) -> ScipyDistributionFunction:
        # Equivalent to adding independent normal random variables
        if not pdfs:
            raise ValueError("No PDFs provided for combination.")

        for pdf in pdfs:
            assert isinstance(pdf, ScipyDistributionFunction)
            assert pdf.short_name == "normal", f"Expected normal distribution, got {pdf.short_name}"

        new_mean = sum([pdf.mean for pdf in pdfs])
        new_variance = sum([pdf.variance for pdf in pdfs])
        return ScipyDistributionFunction(norm, loc=new_mean, scale=new_variance**0.5)
