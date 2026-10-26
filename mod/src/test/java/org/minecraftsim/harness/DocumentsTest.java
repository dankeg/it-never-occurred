package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.List;
import net.minecraft.core.component.DataComponents;
import net.minecraft.network.chat.Component;
import net.minecraft.server.network.Filterable;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.component.WritableBookContent;
import net.minecraft.world.item.component.WrittenBookContent;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins contract-book page parsing, raw JSON escaping and document revision encodings. */
final class DocumentsTest {
	/** Initializes registries once for written and writable book components. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Page arguments preserve ordered strings and leave contract JSON available for exact parsing. */
	@Test
	void pageArgumentsPreserveContractJsonAndPageOrder() {
		JsonObject args = JsonParser.parseString("{\"pages\":[\"{\\\"revision\\\":1,\\\"promise\\\":\\\"deliver 2 bread\\\"}\",\"second\\npage\"]}")
			.getAsJsonObject();
		List<String> pages = Documents.strings(args, "pages", -1);
		assertEquals(List.of("{\"revision\":1,\"promise\":\"deliver 2 bread\"}", "second\npage"), pages);
		assertEquals(JsonParser.parseString("{\"revision\":1,\"promise\":\"deliver 2 bread\"}"), JsonParser.parseString(pages.getFirst()));
	}

	/** Missing page arrays and nonstring entries return the exact document argument errors. */
	@Test
	void pageArgumentsRejectInvalidArraysAndEntriesWithExactErrors() {
		for (String invalid : new String[]{"{}", "{\"pages\":null}", "{\"pages\":{}}", "{\"pages\":\"text\"}"}) {
			JsonObject args = JsonParser.parseString(invalid).getAsJsonObject();
			assertEquals("pages must be an array of strings",
				assertThrows(IllegalArgumentException.class, () -> Documents.strings(args, "pages", -1)).getMessage());
		}
		for (String invalid : new String[]{"1", "true", "null", "{}", "[]"}) {
			JsonObject args = JsonParser.parseString("{\"pages\":[" + invalid + "]}").getAsJsonObject();
			assertEquals("pages entry must be a string",
				assertThrows(IllegalArgumentException.class, () -> Documents.strings(args, "pages", -1)).getMessage());
		}
		assertEquals(List.of(), Documents.strings(JsonParser.parseString("{\"pages\":[]}").getAsJsonObject(), "pages", -1));
	}

	/** Written books retain title, author, contract revisions, page order and exact escaped text. */
	@Test
	void writtenBookContentsPreserveContractJsonAndEscaping() {
		String contract = "{\"revision\":2,\"promise\":\"deliver 2 bread\"}";
		String escaped = "A\"B\\C\n\t";
		ItemStack stack = new ItemStack(Items.WRITTEN_BOOK);
		stack.set(DataComponents.WRITTEN_BOOK_CONTENT, new WrittenBookContent(Filterable.passThrough("Contract \"A\""),
			"alice", 0, List.of(Filterable.<Component>passThrough(Component.literal(contract)),
				Filterable.<Component>passThrough(Component.literal(escaped))), true));
		JsonObject actual = Documents.bookContents(stack);
		JsonObject expected = new JsonObject();
		expected.addProperty("title", "Contract \"A\"");
		expected.addProperty("author", "alice");
		JsonArray pages = new JsonArray();
		pages.add(contract);
		pages.add(escaped);
		expected.add("pages", pages);
		assertEquals(expected, actual);
		assertEquals("\"A\\\"B\\\\C\\n\\t\"", actual.getAsJsonArray("pages").get(1).toString());
		assertEquals(JsonParser.parseString(contract), JsonParser.parseString(actual.getAsJsonArray("pages").get(0).getAsString()));
		assertEquals(expected, JsonParser.parseString(actual.toString()));
	}

	/** Writable books return only their raw ordered pages without adding written-book metadata. */
	@Test
	void writableBookContentsPreserveRawPagesWithoutTitleOrAuthor() {
		ItemStack stack = new ItemStack(Items.WRITABLE_BOOK);
		stack.set(DataComponents.WRITABLE_BOOK_CONTENT,
			new WritableBookContent(List.of(Filterable.passThrough("{\"revision\":1}"), Filterable.passThrough("second\npage"))));
		assertEquals(JsonParser.parseString("{\"pages\":[\"{\\\"revision\\\":1}\",\"second\\npage\"]}"), Documents.bookContents(stack));
	}

	/** Items without either book-content component fail with the established document error. */
	@Test
	void bookContentsRejectItemsWithoutBookComponents() {
		assertEquals("Item has no book content", assertThrows(IllegalArgumentException.class,
			() -> Documents.bookContents(new ItemStack(Items.STONE))).getMessage());
	}

	/** Revision lookup defaults to zero and preserves independent stored document versions without mutation. */
	@Test
	void documentVersionsDefaultToZeroAndReadStoredRevisionsWithoutMutation() {
		JsonObject versions = JsonParser.parseString("{\"book:42\":1,\"book:43\":7}").getAsJsonObject();
		JsonObject before = versions.deepCopy();
		assertEquals(0L, Documents.version(versions, "book:99"));
		assertEquals(1L, Documents.version(versions, "book:42"));
		assertEquals(7L, Documents.version(versions, "book:43"));
		assertEquals(before, versions);
		versions.addProperty("book:42", 2L);
		assertEquals(2L, Documents.version(versions, "book:42"));
		assertEquals(7L, Documents.version(versions, "book:43"));
	}

	/** Provenance includes the exact document identity and supplied revision with no normalization. */
	@Test
	void provenanceDistinguishesDocumentRevisionsExactly() {
		assertEquals("doc:book:42@v0", Documents.provenance("book:42", 0L));
		assertEquals("doc:book:42@v1", Documents.provenance("book:42", 1L));
		assertEquals("doc:book:42@v2", Documents.provenance("book:42", 2L));
		assertEquals("doc:book:43@v2", Documents.provenance("book:43", 2L));
		assertEquals("doc:sign:1,2,3@v7", Documents.provenance("sign:1,2,3", 7L));
	}
}
