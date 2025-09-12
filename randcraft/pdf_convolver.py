import numpy as np

from randcraft.models import Statistics, sum_uncertain_floats
from randcraft.pdfs.anonymous import AnonymousDistributionFunction
from randcraft.pdfs.base import (
    ProbabilityDistributionFunction,
    T_Pdf,
)
from randcraft.pdfs.discrete import DiracDeltaDistributionFunction, DiscreteDistributionFunction
from randcraft.pdfs.mixture import MixtureDistributionFunction
from randcraft.pdfs.normal import NormalDistributionFunction


class PdfConvolver:
    @classmethod
    def convolve_pdfs(cls, pdfs: list[ProbabilityDistributionFunction]) -> ProbabilityDistributionFunction:
        assert len(pdfs) >= 2, "At least two PDFs are required for combination."
        if not all(isinstance(pdf, ProbabilityDistributionFunction) for pdf in pdfs):
            types = {pdf.__class__.__name__ for pdf in pdfs}
            raise TypeError(f"All PDFs must be instances of ProbabilityDistributionFunction, got: {types}")

        if all(isinstance(rv, NormalDistributionFunction) for rv in pdfs):
            return cls.convolve_normals(pdfs=pdfs)  # type: ignore

        discrete_pdfs = [pdf for pdf in pdfs if isinstance(pdf, DiscreteDistributionFunction)]
        continuous_pdfs = [pdf for pdf in pdfs if not isinstance(pdf, DiscreteDistributionFunction)]

        if len(discrete_pdfs) >= 2:
            # First we convolve together all discrete pdfs
            reduced_discrete = cls.convolve_discretes(pdfs=discrete_pdfs)

            if not len(continuous_pdfs):
                return reduced_discrete

            new_pdfs = [reduced_discrete, *continuous_pdfs]
            return cls.convolve_pdfs(pdfs=new_pdfs)

        if len(discrete_pdfs) == 1 and len(continuous_pdfs) == 1:
            return cls.convolve_with_discrete(pdf_a=continuous_pdfs[0], discrete=discrete_pdfs[0])

        return cls.convolve_anon(pdfs=pdfs)

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
        return result_pdf

    @classmethod
    def convolve_normals(cls, pdfs: list[NormalDistributionFunction]) -> NormalDistributionFunction:
        # Equivalent to adding independent normal random variables
        if not pdfs:
            raise ValueError("No PDFs provided for combination.")

        for pdf in pdfs:
            assert isinstance(pdf, NormalDistributionFunction)

        new_mean = sum([pdf.mean for pdf in pdfs])
        new_variance = sum([pdf.variance for pdf in pdfs])
        return NormalDistributionFunction(mean=new_mean, std_dev=new_variance**0.5)

    @classmethod
    def convolve_with_discrete(
        cls, pdf_a: T_Pdf, discrete: DiscreteDistributionFunction
    ) -> T_Pdf | MixtureDistributionFunction:
        assert isinstance(discrete, DiscreteDistributionFunction)

        # Shortcut for dirac delta
        if isinstance(discrete, DiracDeltaDistributionFunction) or len(discrete.values) == 1:
            return pdf_a.add_constant(x=discrete.values[0])

        pdfs = [pdf_a.add_constant(di) for di in discrete.values]
        probabilities = discrete.probabilities
        return MixtureDistributionFunction(pdfs=pdfs, probabilities=probabilities)

    @classmethod
    def convolve_anon(cls, pdfs: list[ProbabilityDistributionFunction]) -> AnonymousDistributionFunction:
        if not pdfs:
            raise ValueError("No PDFs provided for combination.")

        for pdf in pdfs:
            assert isinstance(pdf, ProbabilityDistributionFunction)

        sum_unc = sum_uncertain_floats

        stats = [pdf.statistics for pdf in pdfs]
        mean = sum_unc([s.mean for s in stats])
        variance = sum_unc([s.variance for s in stats])
        second_moment = variance + mean.apply(lambda x: x**2)
        # TODO calculate more moments

        min_value = sum_unc([s.min_value for s in stats])
        max_value = sum_unc([s.max_value for s in stats])

        statistics = Statistics(moments=[mean, second_moment], support=(min_value, max_value))

        def sampler(n: int) -> np.ndarray:
            samples = [p.sample_numpy(n) for p in pdfs]
            return np.sum(samples, axis=0)

        return AnonymousDistributionFunction(sampler=sampler, n_samples=5000, external_statistics=statistics)
