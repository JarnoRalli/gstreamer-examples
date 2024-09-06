import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def create_element(gst_elem: str, name: str):
    """
    Creates a Gst element. Gst.init() has to be called before using this function.

    :param gst_elem: type of the Gst-element, e.g. filesrc
    :param name: name given to the Gst-element
    :return: created element
    """

    # Create an empty element
    new_element = None

    # Try creating an element
    print(f"Creating element: {gst_elem}")
    new_element = Gst.ElementFactory.make(gst_elem, name)
    assert new_element is not None

    return new_element


def link_elements(elements: list) -> None:
    """
    Links a list of Gst.Element elements.

    :param elements: a list of Gst-elements that are to be linked
    :return: None
    """
    assert isinstance(elements, list), "'elements' must be of type list"
    assert len(elements) >= 2

    for idx, x in enumerate(elements[:-1]):
        print(
            f"Linking element {elements[idx].get_name()} -> {elements[idx + 1].get_name()}...",
            end="",
        )
        assert isinstance(
            elements[idx], Gst.Element
        ), "elements[idx] must be of type Gst.Element"
        assert isinstance(
            elements[idx + 1], Gst.Element
        ), "elements[idx+1] must be of type Gst.Element"
        if elements[idx].link(elements[idx + 1]):
            print("done")
        else:
            print("failed")
            raise RuntimeError(
                f"Failed to link: {elements[idx].get_name()} -> {elements[idx + 1].get_name()}"
            )


class PadAddedLinkFunctor:
    """
    A functor that can be used with pad-added messages to dynamically link new pads with subsequent
    elements' sink pads. Before using, an instance of the PadAddedLinkFunctor registers information
    regarding which new pads will be linked to which element.
    """

    def __init__(self):
        self.connections = []

    def register(
        self, new_pad: str, target_element: Gst.Element, target_sink_name: str
    ) -> None:
        """
        Registers linking information indicating how new pads should be linked to subsequent elements.

        In the following example demuxer's video_0 pad will be linked to parser's sink pad.

        demuxer = Gst.ElementFactory("qtdemux", "qtdemuxer")
        parser = Gst.ElementFactory("h264parse", "h264parser")
        pad_added_functor = PadAddedLinkFunctor()
        pad_added_functor.register("video_", parser, , "sink")
        demuxer.connect("pad-added", pad_added_functor)

        :param new_pad: name of the new pad that is linked: new_pad -> target_element.target_sink_name
        :param target_element: target gst-element
        :param target_sink_name: name of the target gst-element sink
        :return: None
        """

        assert isinstance(new_pad, str), "'new_pad' must be of type str"
        assert isinstance(
            target_element, Gst.Element
        ), "'target_element' must be of type Gst.Element"
        assert isinstance(
            target_sink_name, str
        ), "'target_sink_name' must be of type str"

        self.connections.append((new_pad, target_element, target_sink_name))

    def __call__(self, element: Gst.Element, pad: Gst.Pad) -> None:
        """
        Functor for pad-added signal.

        :param element: Gst.Element that had created a new pad
        :param pad: Gst.Pad that has been created
        :return: None
        """

        assert isinstance(element, Gst.Element), "'element' must be of type Gst.Element"
        assert isinstance(pad, Gst.Pad), "'pad' must be of type Gst.Pad"

        pad_name = pad.get_name()
        element_name = element.get_name()

        print(f"New pad '{pad_name}' created")

        # Search if the new pad corresponds to any of the defined connections
        index = [i for i, v in enumerate(self.connections) if pad_name.startswith(v[0])]
        if len(index) == 1:
            index = index.pop()
            target_element = self.connections[index][1]
            target_sink_name = self.connections[index][2]
            sink_pad = target_element.get_static_pad(target_sink_name)

            assert (
                sink_pad
            ), f"'{target_element.get_name()}' has no static pad called '{target_sink_name}'"

            if not sink_pad.is_linked():
                print(
                    f"Linking '{element_name}:{pad_name}' \
                -> '{target_element.get_name()}:{sink_pad.get_name()}'...",
                    end="",
                )
                ret = pad.link(sink_pad)
                if ret == Gst.PadLinkReturn.OK:
                    print("done")
                else:
                    print("error")

        elif len(index) > 1:
            raise RuntimeError(
                f"Pad '{pad_name}' corresponds to several link-definitions, cannot continue"
            )
        else:
            return
