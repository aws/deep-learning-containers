import random


STOP_MAX_DELAY = 40 * 60 * 1000  # 40 min


class RandomExponentialBackoff:
    def __init__(
        self, base=2, wait_exponential_multiplier=1, wait_exponential_max=STOP_MAX_DELAY, deterministic_wait=False
    ):
        """
        Wait in the range of [1, multiplier * (base ** attempt_number)] milliseconds.
        Theory: https://en.wikipedia.org/wiki/Exponential_backoff#Collision_avoidance
        This class can be used with the retrying module from https://github.com/rholder/retrying, by
        using its wait time generator as the wait_func override. The implementation of exponential backoff on the
        retrying module is meant for a "Rate Limiting" use case, rather than a "Collision Avoidance" use case.

        :param base: double Base for exponential wait calculation, default = 2
        :param wait_exponential_multiplier: double Linear multiplier for exponential function result, default = 1
        :param wait_exponential_max: double Maximum wait-time cap to prevent extremely long wait times, default = 2 min
        :param deterministic_wait: bool Set to True if the wait function is meant to be used for Rate Limiting, or any
                                   case where wait time needs to be deterministic, rather than random. default = False
        """
        self.base = base
        self.wait_exponential_multiplier = wait_exponential_multiplier
        self.wait_exponential_max = wait_exponential_max
        self.deterministic_wait = deterministic_wait

    def generate_wait_time_milliseconds(self, attempt_number, delay_since_first_attempt_ms):
        """
        Calculate the maximum possible wait-time for a given attempt_number, and return a random wait-time upto the
        maximum value, so that the backoff for conflicting parallel attempts does not result in more repeated conflicts.
        If the deterministic_wait option for this RandomExponentialBackoff object is True, this function always returns
        the maximum wait-time.
        The function signature includes a second "delay_since_first_attempt_ms" parameter that is unused because
        this is the function signature expected by the wait function, wait_func, on `retrying.retry`.

        :param attempt_number: int Attempt number for which wait-time must be determined, which is used as the power for
                               the exponential function.
        :param delay_since_first_attempt_ms: double Unused dummy input
        :return: double Wait time between previous attempt and upcoming attempt
        """
        wait_time_min = 1  # 1 ms
        wait_time_max = self.wait_exponential_multiplier * (self.base ** attempt_number)
        if self.deterministic_wait:
            return wait_time_max
        return min(random.randint(wait_time_min, wait_time_max), self.wait_exponential_max)
