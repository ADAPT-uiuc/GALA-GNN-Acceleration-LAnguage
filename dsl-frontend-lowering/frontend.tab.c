/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 2 "frontend.y"

#include <iostream>
#include <string.h>
#include "ir/frontendIR.h"
using namespace std;

extern int yylex();
extern int yyparse();
extern FILE *yyin;

void yyerror(const char *s);
FrontendIRNode *root;

#line 85 "frontend.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "frontend.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_IDENTIFIER = 3,                 /* IDENTIFIER  */
  YYSYMBOL_ASSIGN = 4,                     /* ASSIGN  */
  YYSYMBOL_LOAD = 5,                       /* LOAD  */
  YYSYMBOL_LPAREN = 6,                     /* LPAREN  */
  YYSYMBOL_RPAREN = 7,                     /* RPAREN  */
  YYSYMBOL_SEMICOLON = 8,                  /* SEMICOLON  */
  YYSYMBOL_QUOTE = 9,                      /* QUOTE  */
  YYSYMBOL_COMMENT = 10,                   /* COMMENT  */
  YYSYMBOL_MODEL = 11,                     /* MODEL  */
  YYSYMBOL_EVAL = 12,                      /* EVAL  */
  YYSYMBOL_TRAIN = 13,                     /* TRAIN  */
  YYSYMBOL_LAYER = 14,                     /* LAYER  */
  YYSYMBOL_LOSS = 15,                      /* LOSS  */
  YYSYMBOL_OPTIMIZER = 16,                 /* OPTIMIZER  */
  YYSYMBOL_ITERS = 17,                     /* ITERS  */
  YYSYMBOL_VAL_STEP = 18,                  /* VAL_STEP  */
  YYSYMBOL_RELAXNLN = 19,                  /* RELAXNLN  */
  YYSYMBOL_QUANT = 20,                     /* QUANT  */
  YYSYMBOL_GRAPH_ATTR = 21,                /* GRAPH_ATTR  */
  YYSYMBOL_FEAT_ATTR = 22,                 /* FEAT_ATTR  */
  YYSYMBOL_RELU = 23,                      /* RELU  */
  YYSYMBOL_RABBIT_REORDER_OP = 24,         /* RABBIT_REORDER_OP  */
  YYSYMBOL_SAMPLE_RANDOM_OP = 25,          /* SAMPLE_RANDOM_OP  */
  YYSYMBOL_COLTILE = 26,                   /* COLTILE  */
  YYSYMBOL_INTEGER = 27,                   /* INTEGER  */
  YYSYMBOL_FLOAT = 28,                     /* FLOAT  */
  YYSYMBOL_LBRACE = 29,                    /* LBRACE  */
  YYSYMBOL_RBRACE = 30,                    /* RBRACE  */
  YYSYMBOL_LSQBRA = 31,                    /* LSQBRA  */
  YYSYMBOL_RSQBRA = 32,                    /* RSQBRA  */
  YYSYMBOL_DOT = 33,                       /* DOT  */
  YYSYMBOL_COMMA = 34,                     /* COMMA  */
  YYSYMBOL_IF = 35,                        /* IF  */
  YYSYMBOL_ELSE = 36,                      /* ELSE  */
  YYSYMBOL_DO = 37,                        /* DO  */
  YYSYMBOL_WHILE = 38,                     /* WHILE  */
  YYSYMBOL_TRUE = 39,                      /* TRUE  */
  YYSYMBOL_FALSE = 40,                     /* FALSE  */
  YYSYMBOL_NOT = 41,                       /* NOT  */
  YYSYMBOL_AND = 42,                       /* AND  */
  YYSYMBOL_OR = 43,                        /* OR  */
  YYSYMBOL_NOTEQ = 44,                     /* NOTEQ  */
  YYSYMBOL_EQ = 45,                        /* EQ  */
  YYSYMBOL_GREATER = 46,                   /* GREATER  */
  YYSYMBOL_LESS = 47,                      /* LESS  */
  YYSYMBOL_GREATEREQ = 48,                 /* GREATEREQ  */
  YYSYMBOL_LESSEQ = 49,                    /* LESSEQ  */
  YYSYMBOL_PLUS = 50,                      /* PLUS  */
  YYSYMBOL_MINUS = 51,                     /* MINUS  */
  YYSYMBOL_MULTIPLY = 52,                  /* MULTIPLY  */
  YYSYMBOL_DIVIDE = 53,                    /* DIVIDE  */
  YYSYMBOL_FFN = 54,                       /* FFN  */
  YYSYMBOL_DATASET = 55,                   /* DATASET  */
  YYSYMBOL_NONLN = 56,                     /* NONLN  */
  YYSYMBOL_SENSEI_OP = 57,                 /* SENSEI_OP  */
  YYSYMBOL_INT = 58,                       /* INT  */
  YYSYMBOL_NEW = 59,                       /* NEW  */
  YYSYMBOL_NULL_KEY = 60,                  /* NULL_KEY  */
  YYSYMBOL_YYACCEPT = 61,                  /* $accept  */
  YYSYMBOL_dsl_prog = 62,                  /* dsl_prog  */
  YYSYMBOL_dsl_stmnts = 63,                /* dsl_stmnts  */
  YYSYMBOL_dsl_stmnt = 64,                 /* dsl_stmnt  */
  YYSYMBOL_data_stmnt = 65,                /* data_stmnt  */
  YYSYMBOL_layer_defs = 66,                /* layer_defs  */
  YYSYMBOL_layer_def = 67,                 /* layer_def  */
  YYSYMBOL_layer_stmnts = 68,              /* layer_stmnts  */
  YYSYMBOL_layer_stmnt = 69,               /* layer_stmnt  */
  YYSYMBOL_model_defs = 70,                /* model_defs  */
  YYSYMBOL_model_def = 71,                 /* model_def  */
  YYSYMBOL_model_stmnts = 72,              /* model_stmnts  */
  YYSYMBOL_model_stmnt = 73,               /* model_stmnt  */
  YYSYMBOL_params = 74,                    /* params  */
  YYSYMBOL_param = 75,                     /* param  */
  YYSYMBOL_args = 76,                      /* args  */
  YYSYMBOL_arg = 77,                       /* arg  */
  YYSYMBOL_type = 78                       /* type  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   87

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  61
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  18
/* YYNRULES -- Number of rules.  */
#define YYNRULES  40
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  80

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   315


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,    48,    48,    49,    55,    69,    70,    84,    91,   101,
     113,   118,   131,   140,   145,   158,   168,   173,   186,   194,
     199,   212,   255,   256,   269,   276,   284,   285,   298,   304,
     310,   316,   322,   328,   334,   340,   344,   348,   353,   354,
     355
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "IDENTIFIER", "ASSIGN",
  "LOAD", "LPAREN", "RPAREN", "SEMICOLON", "QUOTE", "COMMENT", "MODEL",
  "EVAL", "TRAIN", "LAYER", "LOSS", "OPTIMIZER", "ITERS", "VAL_STEP",
  "RELAXNLN", "QUANT", "GRAPH_ATTR", "FEAT_ATTR", "RELU",
  "RABBIT_REORDER_OP", "SAMPLE_RANDOM_OP", "COLTILE", "INTEGER", "FLOAT",
  "LBRACE", "RBRACE", "LSQBRA", "RSQBRA", "DOT", "COMMA", "IF", "ELSE",
  "DO", "WHILE", "TRUE", "FALSE", "NOT", "AND", "OR", "NOTEQ", "EQ",
  "GREATER", "LESS", "GREATEREQ", "LESSEQ", "PLUS", "MINUS", "MULTIPLY",
  "DIVIDE", "FFN", "DATASET", "NONLN", "SENSEI_OP", "INT", "NEW",
  "NULL_KEY", "$accept", "dsl_prog", "dsl_stmnts", "dsl_stmnt",
  "data_stmnt", "layer_defs", "layer_def", "layer_stmnts", "layer_stmnt",
  "model_defs", "model_def", "model_stmnts", "model_stmnt", "params",
  "param", "args", "arg", "type", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-24)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-3)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int8 yypact[] =
{
      11,    12,     5,   -24,     9,   -24,   -24,    16,   -24,     2,
      10,   -24,    20,   -24,    22,    23,   -24,     7,    27,   -24,
      11,    29,    24,   -24,    28,     6,   -24,    31,   -24,    33,
      -6,   -24,    32,    34,    30,    13,   -24,   -24,   -24,   -24,
      37,    -2,   -24,    36,   -24,    14,    15,    -3,    38,     0,
     -24,    42,    17,    39,    21,    26,    35,    40,   -24,   -24,
      54,   -24,    57,    41,    42,   -24,   -24,   -24,   -24,   -24,
     -24,    58,    59,   -24,   -24,    60,   -24,    -1,    56,   -24
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       5,     0,     3,     1,     0,     8,     6,     0,    10,     0,
       0,    11,     0,    16,     0,     0,     7,     0,     0,    17,
       5,    19,     0,    22,     0,     0,     4,     0,    20,     0,
       0,    22,     0,     0,     0,     0,    38,    39,    40,    23,
       0,     0,    26,     0,     5,    25,     0,     0,     0,     0,
      24,     0,    29,     0,    37,    31,    33,    35,    27,     9,
       0,    12,     0,     0,    13,    28,    21,    36,    30,    32,
      34,     0,     0,    18,    14,     0,    26,     0,     0,    15
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -24,    45,    43,   -24,   -24,   -24,    61,   -24,   -23,   -24,
      55,   -24,    49,    44,   -24,    -4,   -24,   -24
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
       0,     1,     2,     6,    16,     7,     8,    63,    64,    12,
      13,    20,    21,    30,    39,    47,    58,    40
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int8 yytable[] =
{
      52,    35,    52,    60,    53,    46,    78,    14,     4,    32,
       5,    -2,     3,     9,    17,     5,    15,    24,    24,    10,
      54,    15,    54,    18,    55,    56,    55,    56,    22,    23,
      61,    25,    27,    29,    31,    33,    34,    32,    42,    43,
      45,    74,    44,    48,    51,    62,    59,    66,    50,    36,
      37,    65,    38,    36,    37,    67,    38,    57,    71,    57,
      68,    72,    75,    14,    79,    26,    76,    19,    11,    69,
      28,    73,    77,     0,    70,    41,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49
};

static const yytype_int8 yycheck[] =
{
       3,     7,     3,     3,     7,     7,     7,     5,     3,     3,
      10,     0,     0,     4,     4,    10,    14,    11,    11,     3,
      23,    14,    23,     3,    27,    28,    27,    28,     6,     6,
      30,     4,     3,     9,     6,     4,     3,     3,     6,     9,
       3,    64,    29,     7,    29,     3,     8,     8,    34,    55,
      56,    34,    58,    55,    56,    34,    58,    60,     4,    60,
      34,     4,     3,     5,     8,    20,     6,    12,     7,    34,
      21,    30,    76,    -1,    34,    31,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    44
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    62,    63,     0,     3,    10,    64,    66,    67,     4,
       3,    67,    70,    71,     5,    14,    65,     4,     3,    71,
      72,    73,     6,     6,    11,     4,    62,     3,    73,     9,
      74,     6,     3,     4,     3,     7,    55,    56,    58,    75,
      78,    74,     6,     9,    29,     3,     7,    76,     7,    63,
      34,    29,     3,     7,    23,    27,    28,    60,    77,     8,
       3,    30,     3,    68,    69,    34,     8,    34,    34,    34,
      34,     4,     4,    30,    69,     3,     6,    76,     7,     8
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    61,    62,    62,    62,    63,    63,    64,    64,    65,
      66,    66,    67,    68,    68,    69,    70,    70,    71,    72,
      72,    73,    74,    74,    75,    75,    76,    76,    77,    77,
      77,    77,    77,    77,    77,    77,    77,    77,    78,    78,
      78
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     0,     1,     5,     0,     2,     3,     1,     7,
       1,     2,     9,     1,     2,     7,     1,     2,     9,     1,
       2,     7,     0,     2,     3,     2,     0,     2,     2,     1,
       2,     1,     2,     1,     2,     1,     2,     1,     1,     1,
       1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* dsl_prog: %empty  */
#line 48 "frontend.y"
           { (yyval.irNode) = NULL; }
#line 1204 "frontend.tab.c"
    break;

  case 3: /* dsl_prog: dsl_stmnts  */
#line 50 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("dsl_prog");
        (yyval.irNode)->addChild((yyvsp[0].irNode));
        root = (yyval.irNode);
    }
#line 1214 "frontend.tab.c"
    break;

  case 4: /* dsl_prog: dsl_stmnts layer_defs model_defs model_stmnts dsl_prog  */
#line 56 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("dsl_prog");
        (yyval.irNode)->addChild((yyvsp[-4].irNode));
        (yyval.irNode)->addChild((yyvsp[-3].irNode));
        (yyval.irNode)->addChild((yyvsp[-2].irNode));
        (yyval.irNode)->addChild((yyvsp[-1].irNode));
        // $$ -> addChild($5) probably not using extra dsl_stmnts so avoid memory bugs 
        root = (yyval.irNode);
    }
#line 1228 "frontend.tab.c"
    break;

  case 5: /* dsl_stmnts: %empty  */
#line 69 "frontend.y"
             { (yyval.irNode) = NULL; }
#line 1234 "frontend.tab.c"
    break;

  case 6: /* dsl_stmnts: dsl_stmnts dsl_stmnt  */
#line 71 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("dsl_stmnts");
        if ((yyvsp[-1].irNode)){
            for (auto child : (yyvsp[-1].irNode)->children){
                (yyval.irNode)->addChild(child);
            }
            // issue is when deleting node, it deletes all children, meaning the child just 
            // added above is also deleted, so invalid read and invalid delete?
            free((yyvsp[-1].irNode));
        }
        (yyval.irNode)->addChild((yyvsp[0].irNode));
    }
#line 1251 "frontend.tab.c"
    break;

  case 7: /* dsl_stmnt: IDENTIFIER ASSIGN data_stmnt  */
#line 85 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("data_stmnt");
        (yyval.irNode)->addParam((yyvsp[-2].sval));
        (yyval.irNode)->addChild((yyvsp[0].irNode));
        free((yyvsp[-2].sval)); // using free instead of delete b/c don't want to delete children nodes
    }
#line 1262 "frontend.tab.c"
    break;

  case 8: /* dsl_stmnt: COMMENT  */
#line 91 "frontend.y"
              { (yyval.irNode) = new FrontendIRNode("comment"); (yyval.irNode)->addParam((yyvsp[0].sval)); free((yyvsp[0].sval)); }
#line 1268 "frontend.tab.c"
    break;

  case 9: /* data_stmnt: LOAD LPAREN QUOTE IDENTIFIER QUOTE RPAREN SEMICOLON  */
#line 102 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("load");
        (yyval.irNode)->addParam((yyvsp[-3].sval));
        free((yyvsp[-3].sval));
    }
#line 1278 "frontend.tab.c"
    break;

  case 10: /* layer_defs: layer_def  */
#line 114 "frontend.y"
    { 
        (yyval.irNode) = new FrontendIRNode("layer_defs");
        (yyval.irNode)->addChild((yyvsp[0].irNode));
    }
#line 1287 "frontend.tab.c"
    break;

  case 11: /* layer_defs: layer_defs layer_def  */
#line 119 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("layer_defs");
        if ((yyvsp[-1].irNode)){
            for (auto child : (yyvsp[-1].irNode)->children){
                (yyval.irNode)->addChild(child);
            }
            free((yyvsp[-1].irNode));
        }
        (yyval.irNode)->addChild((yyvsp[0].irNode));
    }
#line 1302 "frontend.tab.c"
    break;

  case 12: /* layer_def: IDENTIFIER ASSIGN LAYER LPAREN params RPAREN LBRACE dsl_stmnts RBRACE  */
#line 132 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("layer_def");
        (yyval.irNode)->addParam((yyvsp[-8].sval));
        if ((yyvsp[-4].irNode)) (yyval.irNode)->addChild((yyvsp[-4].irNode)); // L1 = layer(args), in this case args is a parameter, might change
        if ((yyvsp[-1].irNode)) (yyval.irNode)->addChild((yyvsp[-1].irNode)); // dsl_stmnts are optional for now, necessary later
        free((yyvsp[-8].sval));
    }
#line 1314 "frontend.tab.c"
    break;

  case 13: /* layer_stmnts: layer_stmnt  */
#line 141 "frontend.y"
    { 
        (yyval.irNode) = new FrontendIRNode("layer_stmnts");
        (yyval.irNode)->addChild((yyvsp[0].irNode));
    }
#line 1323 "frontend.tab.c"
    break;

  case 14: /* layer_stmnts: layer_stmnt layer_stmnt  */
#line 146 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("layer_stmnts");
        if ((yyvsp[-1].irNode)){
            for (auto child : (yyvsp[-1].irNode)->children){
                (yyval.irNode)->addChild(child);
            }
            free((yyvsp[-1].irNode));
        }
        (yyval.irNode)->addChild((yyvsp[0].irNode));
    }
#line 1338 "frontend.tab.c"
    break;

  case 15: /* layer_stmnt: IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON  */
#line 159 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("layer_stmnt");
        (yyval.irNode)->addParam((yyvsp[-6].sval));
        (yyval.irNode)->addParam((yyvsp[-4].sval));
        if ((yyvsp[-2].irNode)) (yyval.irNode)->addChild((yyvsp[-2].irNode)); // args
        
        free((yyvsp[-6].sval));
    }
#line 1351 "frontend.tab.c"
    break;

  case 16: /* model_defs: model_def  */
#line 169 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("model_defs");
        (yyval.irNode)->addChild((yyvsp[0].irNode));
    }
#line 1360 "frontend.tab.c"
    break;

  case 17: /* model_defs: model_defs model_def  */
#line 174 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("model_defs");
        if ((yyvsp[-1].irNode)){
            for (auto child : (yyvsp[-1].irNode)->children){
                (yyval.irNode)->addChild(child);
            }
            free((yyvsp[-1].irNode));
        }
        (yyval.irNode)->addChild((yyvsp[0].irNode));
    }
#line 1375 "frontend.tab.c"
    break;

  case 18: /* model_def: IDENTIFIER ASSIGN MODEL LPAREN params RPAREN LBRACE layer_stmnts RBRACE  */
#line 187 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("model_def");
        (yyval.irNode)->addParam((yyvsp[-8].sval));
        if ((yyvsp[-4].irNode)) (yyval.irNode)->addChild((yyvsp[-4].irNode)); 
        if ((yyvsp[-1].irNode)) (yyval.irNode)->addChild((yyvsp[-1].irNode)); 
    }
#line 1386 "frontend.tab.c"
    break;

  case 19: /* model_stmnts: model_stmnt  */
#line 195 "frontend.y"
    { 
        (yyval.irNode) = new FrontendIRNode("model_stmnts");
        (yyval.irNode)->addChild((yyvsp[0].irNode));
    }
#line 1395 "frontend.tab.c"
    break;

  case 20: /* model_stmnts: model_stmnt model_stmnt  */
#line 200 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("model_stmnts");
        if ((yyvsp[-1].irNode)){
            for (auto child : (yyvsp[-1].irNode)->children){
                (yyval.irNode)->addChild(child);
            }
            free((yyvsp[-1].irNode));
        }
        (yyval.irNode)->addChild((yyvsp[0].irNode));
    }
#line 1410 "frontend.tab.c"
    break;

  case 21: /* model_stmnt: IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON  */
#line 213 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("model_stmnt");
        (yyval.irNode)->addParam((yyvsp[-6].sval));
        (yyval.irNode)->addParam((yyvsp[-4].sval));
        if ((yyvsp[-2].irNode)) (yyval.irNode)->addChild((yyvsp[-2].irNode)); // args
        
        free((yyvsp[-6].sval));
    }
#line 1423 "frontend.tab.c"
    break;

  case 22: /* params: %empty  */
#line 255 "frontend.y"
         { (yyval.irNode) = NULL; }
#line 1429 "frontend.tab.c"
    break;

  case 23: /* params: params param  */
#line 257 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("params");
        if ((yyvsp[-1].irNode)){
            for (string arg : (yyvsp[-1].irNode)->params)
                (yyval.irNode)->addParam(arg);
            delete (yyvsp[-1].irNode);
        }
        
        (yyval.irNode)->addParam((yyvsp[0].sval));
        free((yyvsp[0].sval));
    }
#line 1445 "frontend.tab.c"
    break;

  case 24: /* param: type IDENTIFIER COMMA  */
#line 270 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[-2].sval)) + strlen((yyvsp[-1].sval)) + 2);
        sprintf((yyval.sval), "%s %s", (yyvsp[-2].sval), (yyvsp[-1].sval));
        free((yyvsp[-2].sval));
        free((yyvsp[-1].sval));
    }
#line 1456 "frontend.tab.c"
    break;

  case 25: /* param: type IDENTIFIER  */
#line 277 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[-1].sval)) + strlen((yyvsp[0].sval)) + 2);
        sprintf((yyval.sval), "%s %s", (yyvsp[-1].sval), (yyvsp[0].sval));
        free((yyvsp[-1].sval));
        free((yyvsp[0].sval));
    }
#line 1467 "frontend.tab.c"
    break;

  case 26: /* args: %empty  */
#line 284 "frontend.y"
       { (yyval.irNode) = NULL; }
#line 1473 "frontend.tab.c"
    break;

  case 27: /* args: args arg  */
#line 286 "frontend.y"
    {
        (yyval.irNode) = new FrontendIRNode("args");
        if ((yyvsp[-1].irNode)){
            for (string arg : (yyvsp[-1].irNode)->params)
                (yyval.irNode)->addParam(arg);
            delete (yyvsp[-1].irNode);
        }
        
        (yyval.irNode)->addParam((yyvsp[0].sval));
        free((yyvsp[0].sval));
    }
#line 1489 "frontend.tab.c"
    break;

  case 28: /* arg: IDENTIFIER COMMA  */
#line 299 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[-1].sval)) + 2);
        sprintf((yyval.sval), "%s", (yyvsp[-1].sval));
        free((yyvsp[-1].sval));
    }
#line 1499 "frontend.tab.c"
    break;

  case 29: /* arg: IDENTIFIER  */
#line 305 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[0].sval)) + 2);
        sprintf((yyval.sval), "%s", (yyvsp[0].sval));
        free((yyvsp[0].sval));
    }
#line 1509 "frontend.tab.c"
    break;

  case 30: /* arg: INTEGER COMMA  */
#line 311 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[-1].sval)) + 2);
        sprintf((yyval.sval), "%s", (yyvsp[-1].sval));
        free((yyvsp[-1].sval));
    }
#line 1519 "frontend.tab.c"
    break;

  case 31: /* arg: INTEGER  */
#line 317 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[0].sval)) + 2);
        sprintf((yyval.sval), "%s", (yyvsp[0].sval));
        free((yyvsp[0].sval));
    }
#line 1529 "frontend.tab.c"
    break;

  case 32: /* arg: FLOAT COMMA  */
#line 323 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[-1].sval)) + 2);
        sprintf((yyval.sval), "%s", (yyvsp[-1].sval));
        free((yyvsp[-1].sval));
    }
#line 1539 "frontend.tab.c"
    break;

  case 33: /* arg: FLOAT  */
#line 329 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[0].sval)) + 2);
        sprintf((yyval.sval), "%s", (yyvsp[0].sval));
        free((yyvsp[0].sval));
    }
#line 1549 "frontend.tab.c"
    break;

  case 34: /* arg: NULL_KEY COMMA  */
#line 335 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[-1].sval)) + 2);
        sprintf((yyval.sval), "%s", (yyvsp[-1].sval));
        free((yyvsp[-1].sval));
    }
#line 1559 "frontend.tab.c"
    break;

  case 35: /* arg: NULL_KEY  */
#line 341 "frontend.y"
    {
        (yyval.sval) = strdup("NULL");
    }
#line 1567 "frontend.tab.c"
    break;

  case 36: /* arg: RELU COMMA  */
#line 345 "frontend.y"
    {
        (yyval.sval) = strdup("dsl.nln.ReLU");
    }
#line 1575 "frontend.tab.c"
    break;

  case 37: /* arg: RELU  */
#line 349 "frontend.y"
    {
        (yyval.sval) = strdup("dsl.nln.ReLU");
    }
#line 1583 "frontend.tab.c"
    break;

  case 38: /* type: DATASET  */
#line 353 "frontend.y"
               { (yyval.sval) = strdup("DSL_Dataset"); }
#line 1589 "frontend.tab.c"
    break;

  case 39: /* type: NONLN  */
#line 354 "frontend.y"
            { (yyval.sval) = strdup("NonLn"); }
#line 1595 "frontend.tab.c"
    break;

  case 40: /* type: INT  */
#line 355 "frontend.y"
          { (yyval.sval) = strdup("int"); }
#line 1601 "frontend.tab.c"
    break;


#line 1605 "frontend.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 358 "frontend.y"


/* C Code */
int main(int, char**) {
    /* read from file instead of stdin */
    FILE *myfile = fopen("input.txt", "r");
    if (!myfile) {
        cout << "Invalid File" << endl;
        return -1;
    }
    yyin = myfile;
    yyparse();
    cout << "------- generating parse tree -----\n";

    int directChildren = FrontendIRNode::countDirectChildren(root);
    vector<bool> flag(directChildren, true); 
    FrontendIRNode::printParseTree(root, std::cout, flag);
    /* FrontendIRNode::generateIR(root, cout); */
    delete root;

    fclose(myfile);
}
void yyerror(const char *s){
    printf("Failed to parse: %s\n", s);
    /* halt program */
    exit(-1);
}
