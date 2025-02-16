from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
)


SorterCallable = Callable[[Sequence], List]
class VisitorCallable(Protocol):
    def __call__(self, k: Any, v: Any, **kwargs: Any) -> Optional[Tuple[Any, Any]]: ...  # pragma: no cover


# [not-sequence-types]
# These python types are not considered to be Sequences in travered
# even though, technically, they are Sequences in Python.
STR_TYPES = (str, bytes, bytearray)
# [not-sequence-types]


def ismapping(arg: Any) -> bool:
    """Check that given object is a Mapping.

    Args:
        arg (Any): Object to check.

    Returns:
        bool: True if arg is a Mapping.
    """

    return isinstance(arg, Mapping)


def issequence(arg: Any) -> bool:
    """Check that given object is a non-string sequence.

    Here, the sequence is an iterable that is not a string,
    sequence of bytes or bytearray.

    Args:
        arg (Any): Object to check.

    Returns:
        bool: True if arg is a sequence.
    """

    return isinstance(arg, Sequence) and not isinstance(arg, STR_TYPES)


def iskvtuple(arg: Optional[Tuple[Any, ...]]) -> bool:
    """Check that given object is a tuple of key and value.

    Args:
        arg: Object to check.

    Returns:
        bool: True if arg is a tuple of length 2.
    """

    return isinstance(arg, tuple) and len(arg) == 2


def default_visitor(k: Any, v: Any, **kwargs: Any) -> Optional[Tuple[Any, Any]]:
    """Default visitor function for traverse.

    Args:
        k (Any): Key of the current mapping.
        v (Any): Value of the current mapping.
        **kwargs: Arbitrary arguments passed to the visitor function.

    Returns:
        Optional[Tuple[Any, Any]]: Tuple of key and value.
    """

    return (k, v)


def default_sorter(items: Sequence) -> List:
    """Default sorter function for traverse.

    Args:
        items (Sequence): Sequence to sort.

    Returns:
        List: Sorted sequence.
    """

    return list(items)


def traverse(
    obj: Any,
    visitor: Optional[VisitorCallable] = None,
    key_sorter: Optional[SorterCallable] = None,
    list_sorter: Optional[SorterCallable] = None,
    repack_into: Type[Dict[Hashable, Any]] = dict,
    **kwargs: Any,
) -> Any:
    """Traverse any object recursively and returns its deep clone.

    Args:
        obj (Any): Object to traverse.
        visitor (Callable):  Function which is invoked for every key and
            value encountered in the traversed object.
        key_sorter (Callable): Function which sorts mapping keys to control the
            order in which they are visited.
        list_sorter (Callable): Function that sorts iterable items to control
            the order in which they are visited.
        repack_into (class): Type to repack mapping objects into. Defaults
            to ``dict``.
        **kwargs: Arbitrary arguments that are recursively passed to each
            invocation of the ``visitor`` function.

    Returns:
        The deep clone of the given ``obj`` object.
        
        Returns a Dict-like object of ``repack_into`` type if the ``obj``
        itself was a mapping.

        Returns an Iterable-like object if the ``obj`` itself is
        an iterable. The type of the returned iterable stays the same.

        Returns the copy of the ``obj`` itself if it's not a 
        mapping or a sequence.

    Performs a *depth first search* (DFS) through the given ``arg`` object.
    
    Traverse invokes the given ``visitor`` function for keys and values
    encountered during traversal. When traversing mappings
    both key and value are passed into the ``visitor``.
    
    During the traversal of the non-string iterables (lists, tuples, etc) and
    plain values (strinonly the ``value`` is
    passed to the ``visitor`` and ``None`` is passed instead of the key.

    Traverse is a very versatile function created for making deep copies of
    any nested mappings and lists, modifying keys and values of such
    structures on the flight, or collecting information about every key-value
    pair in the structure by visiting all of them. 
    
    A lot of the logic in Mapz relies on this method to do things.
    For example:

        - :py:func:`.map` method relies on traverse to apply
          a given function to every key-value pair of the dictionary.
        - :py:func:`.to_table` modifier uses traverse to explore every key
          and value of the dictionary and build a table out of it,
          row by row.
        - :py:func:`.deepcopy` method uses traverse to clone a dictionary
          with all its nested objects.

    Several additional parameters are passed from traverse to the visitor
    function when it's invoked for every key and value:

        - ``_depth`` indicates the current level of recursion with ``1`` 
          being the starting one.
        - ``_index`` indicates the number of the currently investigated
          element within the container if such element is a part of the list
          or a mapping. Just like any position index in Python it starts
          with ``0``.
        - ``_ancestors`` represents the list of current element's ancestors
          with the last item in the list being the most recent ancestor.
        - additionally, any ``**kwargs`` passed to the traverse are supplied
          to the visitor function. This can be used to utilize visitor
          function as an information or data collector.
    
    ``traverse`` can be used by Mapz to transform ``dict`` objects, however
    nested they might be, into a ``Mapz`` object. That means that every
    nested ``dict`` must be repacked into a different mapping type.
    Thus, the type of the resulting returned object cannot be reliably
    determined from the type of the given ``arg``.
    Using ``type(arg)`` inside the ``traverse``'s implementation would
    result in the returned object being of the same type as the ``arg``.
    So, passing a ``dict`` into the ``traverse`` will result in the ``dict``
    being returned.
    To change the mapping type during traversal a ``mapping_type`` argument
    is used.
    
   """
    if visitor is None:
        visitor = default_visitor

    if key_sorter is None:
        key_sorter = default_sorter

    if list_sorter is None:
        list_sorter = default_sorter

    if "_depth" not in kwargs:
        kwargs["_depth"] = 0
    kwargs["_depth"] += 1

    if "_index" not in kwargs:
        kwargs["_index"] = 0

    if "_ancestors" not in kwargs:
        kwargs["_ancestors"] = []
    else:
        # Have to clone ancestor list here, because otherwise the list
        # will share single instance between all traversals.
        # Instead, it should split into as many copies as needed. Each for
        # its own traversal path.
        kwargs["_ancestors"] = list(kwargs["_ancestors"]) + [obj]

    result: Optional[Tuple[Any, ...]] = None

    if ismapping(obj):
        # This branch always returns mapping
        d = repack_into()

        keys = key_sorter(obj.keys())
        for idx, k in enumerate(keys):
            v = obj[k]

            kwargs["_index"] = idx

            # At this point, ``func``` can transform both ``k`` and ``v``
            # to anything, even to None. Or turn ``v`` into a plain value.
            result = visitor(k, v, **kwargs)
            if result is not None and iskvtuple(result):
                k, v = result

            if ismapping(v) or issequence(v):
                v = traverse(
                    v,
                    visitor,
                    key_sorter=key_sorter,
                    list_sorter=list_sorter,
                    repack_into=repack_into,
                    **kwargs,
                )

            dict.__setitem__(d, k, v)

        return d

    elif issequence(obj):
        # This branch always returns sequence of the same type as ``arg``.
        l: List[Any] = []

        items = list_sorter(obj)
        for idx, i in enumerate(items):

            kwargs["_index"] = idx

            result = visitor(None, i, **kwargs)
            if result is not None and iskvtuple(result):
                k, i = result

            if ismapping(i) or issequence(i):
                i = traverse(
                    i,
                    visitor,
                    key_sorter=key_sorter,
                    list_sorter=list_sorter,
                    repack_into=repack_into,
                    **kwargs,
                )

            l.append(i)

        t = type(obj)
        return t(l)

    else:
        # This branch returns whatever results we get from ``func`` or
        # ``arg`` itself if there were no results.
        result = visitor(None, obj, **kwargs)

        if result is not None and iskvtuple(result):
            k, v = result
            return v
        else:
            return obj
